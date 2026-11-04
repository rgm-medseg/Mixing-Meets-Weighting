"""Complete all currently available historical analyses before formal training."""
import os
import argparse
import subprocess
from common import *

def main():
    p=argparse.ArgumentParser(); p.add_argument('--reuse',nargs='*',choices=['A1','A2','A3','A4','A5','A6']); a=p.parse_args()
    env=os.environ.copy(); env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',WANDB_MODE='disabled')
    outcomes=[]
    commands=[('Historical_baseline',[str(PYTHON),str(WORK/'scripts/evaluate.py'),'--task','Historical_baseline']),
              ('Historical_Full',[str(PYTHON),str(WORK/'scripts/evaluate.py'),'--task','Historical_Full']),
              ('COVID_existing',[str(PYTHON),str(WORK/'scripts/reevaluate_existing_covid.py')])]
    commands += [(f'A{i}',[str(PYTHON),str(WORK/f'scripts/analyze_A{i}.py'),'--pre']) for i in range(1,7)]
    for task,command in commands:
        if task in (a.reuse or []) and (WORK/f'analysis/{task}/status.json').exists():
            outcomes.append(dict(task=task,exit_code=0,log=str(WORK/f'logs/pre_{task}.log'),reused=True)); print(task,'reused completed pre-analysis',flush=True); continue
        with (WORK/f'logs/pre_{task}.log').open('w') as f:
            code=subprocess.run(command,cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        outcomes.append(dict(task=task,exit_code=code,log=str(WORK/f'logs/pre_{task}.log')))
        print(task,code,flush=True)
    atomic_json(WORK/'reports/pre_analysis_execution.json',dict(time=now(),tasks=outcomes))
    descriptions=[]
    for i in range(1,7):
        task=f'A{i}'; status=WORK/f'analysis/{task}/status.json'
        row=read_json(status) if status.exists() else dict(status='BLOCKED',reason='Pre-analysis command failed; see log')
        descriptions.append(f"- {task}: {row['status']} — {row.get('reason','')}\n  依赖补齐：新E1–E5验证选中checkpoint/全步日志/统一test评价；A2/A3无E5 EMA要求。")
    fixed=read_json(WORK/'configs/fixed_scale.json')
    text=f'''# 启动前分析

时间{now()}。所有可用历史材料已执行分析，不补写失落历史轨迹。

Full与baseline历史配置/源码/检查点来源已核验，统一评价在独立runs/Historical_*目录，原稿成绩不覆盖；如有数值差异，以新口径实际结果和来源解释。预训练文件确定，生成历史预算/seed仍NA，不推测。

固定尺度97×97，s={fixed['fixed_side']}，历史每200step的150条面积样本均相同；稀疏覆盖不代表全部训练step，记录s仍变化。E2/E3锁定实际面积误差0，不用mean(s)替代RMS/取整面积准则。原Full动态方案可能几何变化很小，该事实应影响解释。

E1均匀权重按控制模型自己的逐图w均值、参与像素为全mixed网格；stop-gradient与仅student1梯度已通过直接张量检查。E4宽18结构647参数对照TRC641，保留零门控与教师EMA。E5使用官方固定commit/nativeU-Net/loss/AdaMix，所有非原Full差异已在hyperparameter_provenance.md写明。

COVID本地baseline与Full所有5%/10%行历史标mm（n=4），但512→256缩放未更新spacing；原值精确重放并保留，另存正确spacing和回原网格的两种评价。最终XYZ重建轴序正确。其双向ASD与ACDC单向ASD不可混同。外部稿件Table4逐行原始映射本地未定位，仅该审计缺失项BLOCKED，未请求论文或等待写作助手。复制视图collate没有额外扰动；FFT实际交换相应mixed视图低频空间分量，复制输入下内容扰动近乎恒等，不预判为实现错误。

lambda扫描组件均ON，seed42/30000updates/valbest；.05是历史hook默认而非已证实优化出来的最优系数，不更改主值，不把扫描变化当seed方差。

'''+ '\n'.join(descriptions)+'\n\n每个脚本退出码、日志见reports/pre_analysis_execution.json。全部分析脚本现在可独立执行；队列结束后自动补齐，有失败或缺失保留PARTIAL/BLOCKED和原因。\n'
    (WORK/'PRE_ANALYSIS.md').write_text(text)
    with (WORK/'PRE_ANALYSIS.md').open('a') as f:
        f.write('\n## COVID缩放spacing追溯更正\n\n最终evaluator沿axis=2正确重建XYZ，轴序判断已更正。512→256缩放后仍用原spacing是已证实的问题，旧mm标签不代表resize后的正确物理间距。已在独立目录保存固定预测的legacy重放、corrected derived center-grid、corrected native-grid三口径评价，无新增COVID训练。见reports/covid_geometry_audit.csv与covid_reevaluation_macro.csv。外部方法行仍缺失，不套用本地更正。\n')
    subprocess.run([str(PYTHON),str(WORK/'scripts/summarize_results.py')],cwd=WORK,env=env,check=True)
    if any(x['exit_code'] for x in outcomes): raise SystemExit(1)

if __name__=='__main__': main()
