import random
import numpy as np, torch
from scipy.ndimage import zoom, rotate
from dataloaders import dataset as ds

def rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state())

def restore(s):
    random.setstate(s[0])
    np.random.set_state(s[1])
    torch.set_rng_state(s[2])

def seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

def boundary(g):
    b = np.zeros(g.shape, bool)
    a = g[1:] != g[:-1]
    b[1:] |= a
    b[:-1] |= a
    a = g[:, 1:] != g[:, :-1]
    b[:, 1:] |= a
    b[:, :-1] |= a
    return b

def recorded_aug(image, gt):
    meta = {}
    geom = {'kind': 'identity'}
    original_rot = ds.random_rot_flip
    original_angle = ds.random_rotate
    original_cut = ds.cutout_gray
    original_params = ds.transforms.ColorJitter.get_params

    def rot(a, b=None):
        r = np.random.RandomState()
        r.set_state(np.random.get_state())
        k = int(r.randint(0, 4))
        axis = int(r.randint(0, 2))
        geom.update(kind='rot_flip', k=k, axis=axis)
        return original_rot(a, b)

    def angle(a, b):
        r = np.random.RandomState()
        r.set_state(np.random.get_state())
        deg = int(r.randint(-20, 20))
        geom.update(kind='rotate', angle=deg)
        return original_angle(a, b)

    def cut(a, b, **kw):
        before = rng()
        out = original_cut(a, b, **kw)
        after = rng()
        restore(before)
        (_, ones) = original_cut(a, np.ones(b.shape, np.uint8), **kw)
        restore(after)
        meta['occlusion'] = np.array(ones) == 0
        pos = np.argwhere(meta['occlusion'])
        meta['cutout_rectangle'] = [int(pos[:, 0].min()), int(pos[:, 1].min()), int(pos[:, 0].max() + 1), int(pos[:, 1].max() + 1)] if len(pos) else None
        return out

    def params(*args):
        ret = original_params(*args)
        meta['color_jitter'] = [ret[0].tolist(), *[float(x) if x is not None else None for x in ret[1:]]]
        return ret
    ds.random_rot_flip = rot
    ds.random_rotate = angle
    ds.cutout_gray = cut
    ds.transforms.ColorJitter.get_params = staticmethod(params)
    try:
        out = ds.WeakStrongAugment([256, 256])(dict(image=image.copy(), label=gt.copy()))
    finally:
        ds.random_rot_flip = original_rot
        ds.random_rotate = original_angle
        ds.cutout_gray = original_cut
        ds.transforms.ColorJitter.get_params = staticmethod(original_params)

    def transform(a):
        if geom['kind'] == 'rot_flip':
            a = np.flip(np.rot90(a, geom['k']), axis=geom['axis']).copy()
        elif geom['kind'] == 'rotate':
            a = rotate(a, geom['angle'], order=0, reshape=False)
        return zoom(a, (256 / a.shape[0], 256 / a.shape[1]), order=0)
    weak_true = transform(gt)
    anat = transform(boundary(gt).astype(np.uint8)).astype(bool)
    valid = transform(np.ones(gt.shape, np.uint8)).astype(bool)
    assert np.array_equal(weak_true.astype(np.uint8), out['label'].numpy())
    occ = meta['occlusion']
    strong_label = weak_true.copy()
    strong_label[occ] = 0
    assert np.array_equal(strong_label.astype(np.uint8), out['label_strong'].numpy())
    return (out, dict(geometry=geom, cutout_rectangle=meta['cutout_rectangle'], color_jitter=meta.get('color_jitter'), weak_true=weak_true, anat=anat, geom_valid=valid, occlusion=occ))
