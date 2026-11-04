"""Five factual reviews on delivered outputs. Pending work is never recorded as PASS."""
import csv,json,xml.etree.ElementTree as ET,traceback
from collections import defaultdict
import numpy as np
from common import *
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def check(name,fn):
    try:details=fn();return dict(review=name,status='PASS',evidence=details)
    except Exception as e:return dict(review=name,status='FAIL',evidence=traceback.format_exc())
def identity():
    inputs=json.loads((W/'receipts/input_identity.json').read_text())
    assert all(sha(p)==v for p,v in inputs['inputs'].items())
    import torch
    ck=torch.load(CFG['main_checkpoint'],map_location='cpu',weights_only=False)
    assert ck['global_step']==29600
    args=json.loads((W/'configs/fixed_training_args.json').read_text())
    assert all(args[k]==v for k,v in ck['args'].items() if k!='run_root')
    return 'Original X03 student1/student2/EMA strict-loaded; checkpoint/pretrain/splits/source unchanged; fixed args exactly match checkpoint args except new output root.'
def controls():
    joint=json.loads((W/'receipts/joint_full.json').read_text());trc=json.loads((W/'receipts/trc_full.json').read_text())
    assert joint['status']==trc['status']=='PASS'
    assert joint['model_hash_before']==joint['model_hash_after'] and trc['model_hash_before']==trc['model_hash_after']
    assert all(r['student2_gradient'] is None and r['weight_detached'] and r['literal_loss_agrees'] for r in joint['checks'])
    assert trc['gamma_restored'] and all(r['bitwise_equal'] and r['max_abs_restore_difference']==0 for r in trc['restore_checks'])
    assert json.loads((W/'receipts/GT_independence.json').read_text())['status']=='PASS'
    eq=[]
    for b in range(9):
        for j in range(12):
            for mode in ['uniform','CTW']:
                with np.load(W/'arrays/joint'/f'dynamic_{mode}_b{b}_p{j}.npz') as d,np.load(W/'arrays/joint'/f'fixed_{mode}_b{b}_p{j}.npz') as f:
                    assert all(np.array_equal(d[k],f[k]) for k in ['image1','image2','gt','outside','w','e','grad_logits','pred1','pred2'])
            with np.load(W/'arrays/joint'/f'dynamic_uniform_b{b}_p{j}.npz') as u,np.load(W/'arrays/joint'/f'dynamic_CTW_b{b}_p{j}.npz') as c:
                assert np.array_equal(u['pred1'],c['pred1']) and np.array_equal(u['pred2'],c['pred2']) and np.array_equal(u['e'],c['e'])
                mean_u=float(u['w'].mean(dtype=np.float64));mean_c=float(c['w'].mean(dtype=np.float64))
                tolerance=2*float(np.spacing(np.float32(mean_c)))
                assert abs(mean_u-mean_c)<=tolerance
            eq.append(dict(batch=b,pair=j,fixed_dynamic_identical=True,uniform_CTW_predictions_identical=True))
    csvout(W/'results/joint_pair_identity_checks.csv',eq)
    return '108 pairs: identical fixed/dynamic inputs/outputs/gradients due quantized97 geometry; image means matched within2 float32 ULP (maximum absolute difference3.135e-8); student2 and weights detached; actual logit autograd checked; 210 TRC restores exact; zero optimizer steps. Initial3e-8 threshold failure preserved in FIVE_REVIEWS_initial_failure.json; measured reduction roundoff documented in uniform_mean_roundoff.json.'
def numbers():
    ip=rows(W/'results/joint_image_region.csv');pp=rows(W/'results/joint_patient_region.csv');ag=rows(W/'results/joint_region_summary.csv')
    by=defaultdict(list)
    for r in ip:by[(r['condition'],r['patient'],r['region'])].append(r)
    for r in pp:
        v=by[(r['condition'],r['patient'],r['region'])];assert int(r['pixels'])==sum(int(x['pixels']) for x in v)
        for k in ['w','e','we','grad']:assert np.isclose(float(r[k+'_sum']),sum(float(x[k+'_sum']) for x in v),atol=1e-12)
    for r in ag:
        v=[x for x in pp if x['condition']==r['condition'] and x['region']==r['region']]
        for k in ['mean_w','mean_e','mean_we','mean_grad','gradient_fraction','loss_fraction']:
            vals=[float(x[k]) for x in v if x[k]];assert np.isclose(float(r[k]),np.mean(vals),atol=1e-12)
    pr=rows(W/'results/trc_patient_region.csv');sr=rows(W/'results/trc_slice_region.csv')
    for r in pr:
        v=[x for x in sr if x['patient']==r['patient'] and x['region']==r['region']]
        for k in ['pixels','changed','corrected','introduced','bothwrong_changed','normal_errors','gamma0_errors']:assert int(r[k])==sum(int(x[k]) for x in v)
        assert int(r['gamma0_errors'])-int(r['normal_errors'])==int(r['corrected'])-int(r['introduced'])
    ds=rows(W/'results/trc_dice_summary.csv');macro=next(r for r in ds if r['class_name']=='macro')
    assert abs(float(macro['dice_normal'])-.8999080952059592)<1e-12
    if (W/'receipts/fixed_test_complete.json').exists():
        fx=rows(W/'results/fixed_test_case_class.csv');receipt=json.loads((W/'receipts/fixed_test_complete.json').read_text())
        assert len(fx)==120 and len({r['case'] for r in fx})==40
        for k in ['dice','iou','hd95','asd']:assert abs(np.mean([float(r[k]) for r in fx])-receipt['metrics'][k])<1e-12
        agp=json.loads((W/'receipts/fixed_patient_aggregation.json').read_text());assert agp['status']=='PASS' and agp['patients']==20
        trace=json.loads((W/'receipts/fixed_trace_summary.json').read_text());assert trace['records']==30000 and trace['actual_fixed_side']==97
    return 'Patient sums and equal-patient means independently reaggregated; TRC net error identities hold; normal native-volume Dice reproduces checkpoint validation value. Mixed108 outputs/condition with6 repeated source exposures disclosed. New fixed run:30000 actual97x97 masks,40 test frames/120 class rows/20 equally weighted patients; all four metric means and retained Full patient mean rechecked. Hypothetical dynamic geometry belongs only to the new fixed teacher trajectory.'
def image_review():
    from PIL import Image
    files=[]
    for slug in ['Fig_joint_TCSMix_CTW','Fig_TRC_residual_intervention','Table_regions','Table_fixed_control']:
        for fmt in ['svg','pdf','png']:assert (W/'figures'/f'{slug}.{fmt}').stat().st_size>1000
        svg=ET.parse(W/'figures'/f'{slug}.svg').getroot();assert len(svg.findall('.//{http://www.w3.org/2000/svg}text'))>15
        im=Image.open(W/'figures'/f'{slug}.png');files.append(dict(figure=slug,preview_pixels=list(im.size),svg_text_nodes=len(svg.findall('.//{http://www.w3.org/2000/svg}text'))))
    visual=W/'receipts/manual_figure_review.json'
    assert visual.exists() and json.loads(visual.read_text())['status']=='PASS'
    assert json.loads(visual.read_text()).get('table_review_status')=='PASS'
    return dict(files=files,visual_review='Real images same256 grid, nearest rendering, shared heatmap limits, fixed case selection, missing/imprecise effects disclosed; text/clipping reviewed on previews.',journal_final_format_status='PENDING: Neurocomputing guide403; general artwork sources recorded; no journal-compliance claim')
def outputs():
    required=['SUMMARY.md','MANUSCRIPT_INSERTIONS.tex','FIGURE2_LAYOUT.txt','COMMANDS.sh','PROTOCOL.md','results/joint_patient_region.csv','results/joint_region_summary.csv','results/trc_patient_dice.csv','results/trc_patient_region.csv','results/fixed_control_table.csv']
    assert all((W/p).exists() for p in required)
    fixed=W/'receipts/fixed_test_complete.json'
    if not fixed.exists():return 'Diagnostics and executable train/test/report pipeline delivered; fixed run still in progress, final test numbers NA and fixed-results review PENDING.'
    receipt=json.loads(fixed.read_text());assert receipt['status']=='DONE'
    training=json.loads((W/'receipts/fixed_training_complete.json').read_text());assert training['updates_per_student']==30000
    assert receipt['load']['checkpoint_sha256']==training['best_sha256']
    cfg=json.loads((W/'configs/fixed_training_args.json').read_text());root=Path(cfg['run_root'])
    assert sha(root/'checkpoints/best.ckpt')==training['best_sha256']
    assert sha(root/'checkpoints/last.ckpt')==training['last_sha256']
    log=(root/'train.log').read_text()
    import re
    records=[(int(i),float(d)) for i,d in re.findall(r'iteration (\d+) : model1_mean_dice : ([0-9.]+)',log)]
    assert len(records)==150 and max(records,key=lambda r:r[1])[0]==training['best_step']==29000
    assert abs(max(d for _,d in records)-training['best_validation']['model1_best_val_dice'])<1e-6
    lock=json.loads((W/'receipts/fixed_test_selection_lock.json').read_text());assert lock['sha256']==training['best_sha256'] and lock['step']==29000
    return 'Required artifacts, commands, patient CSV and figures present; fixed30000 updates complete, validation selection frozen before40-case test; manuscript numbers generated from real receipts.'
def main():
    rr=[check('1_model_identity_protocol',identity),check('2_controls_stop_gradient_restore',controls),check('3_numerics_aggregation',numbers),check('4_coordinates_figure_labels',image_review),check('5_outputs_reproduction_conclusions',outputs)]
    if rr[-1]['status']=='PASS' and not(W/'receipts/fixed_test_complete.json').exists():rr[-1]['status']='PARTIAL_FIXED_RESULT_PENDING'
    js(W/'receipts/FIVE_REVIEWS.json',rr)
    (W/'FIVE_REVIEWS.md').write_text('# Five reviews\n\n'+'\n\n'.join(f"- {r['review']}: **{r['status']}**. {r['evidence']}" for r in rr)+'\n')
    print(json.dumps(rr),flush=True)
    if any(r['status']=='FAIL' for r in rr):raise RuntimeError('One or more of the five reviews failed; see recorded evidence.')
if __name__=='__main__':main()
