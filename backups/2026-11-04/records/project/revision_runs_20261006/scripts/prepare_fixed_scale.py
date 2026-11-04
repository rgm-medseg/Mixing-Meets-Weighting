"""Lock nearest realizable rectangle area using actual historical Full endpoint logs."""
import json
import math
import re
from common import WORK, FULL, atomic_json, sha, write_csv

def main():
    source = FULL / 'train.log'
    rows = []
    for line in source.read_text().splitlines():
        m = re.search(r'iteration (\d+) : adamix_endpoint=(\{.*\})', line)
        if m:
            p = json.loads(m[2])
            rows.append(dict(step=int(m[1]), q=p['teacher_conf'], s=p['side_fraction'], area=p['patch_area_ratio']))
    if not rows:
        raise RuntimeError('BLOCKED: no historical area source; no extra Full training authorized')
    target = sum(r['area'] for r in rows) / len(rows)
    squared = sum(r['s'] ** 2 for r in rows) / len(rows)
    # Historical max/min/round rules at H=W=256; choose geometry, not test score.
    h = w = 256
    candidates = [(abs((k*k)/(h*w)-target), k) for k in range(1, 256)]
    error, k = min(candidates)
    value = dict(locked=True, source=str(source), source_sha256=sha(source),
                 coverage='sparse historical training endpoints, every 200 updates; not all-step mean',
                 samples=len(rows), first_step=rows[0]['step'], last_step=rows[-1]['step'],
                 observed_mean_actual_area=target, rms_side=math.sqrt(squared), fixed_side=k/256,
                 rectangle=[k,k], fixed_area=k*k/65536, absolute_area_error=error,
                 relative_area_error=error/target, rounding='Python round, clamp [1,H-1]/[1,W-1]',
                 position='historical np.random.randint(0,max(1,H-patch_h)); same for W',
                 selection='nearest representable actual area; no test access')
    out = WORK / 'configs/fixed_scale.json'
    if out.exists() and json.loads(out.read_text()) != value:
        raise RuntimeError('Refusing to overwrite a different locked fixed scale')
    atomic_json(out, value)
    write_csv(WORK/'analysis/A1/historical_full_scale.csv', rows)
    print(json.dumps(value, indent=2))

if __name__ == '__main__':
    main()
