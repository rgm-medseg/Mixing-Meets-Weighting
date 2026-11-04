import shutil,platform,subprocess
from common import *
O=ROOT/'offline_A2_A5_20261007'
def main():
 cfg=read(O/'configs/analysis.json');cfg.update(equivalence_atol=1e-6,equivalence_rtol=1e-5,paths=['original_runtime','deployment_no_diagnostics'],inputs=['zeros','validation_first_slice'],a3_seed=4261007,a3_widths=[2,1,3],quantiles=[0,.2,.4,.6,.8,1],a3_modes=['eval'],a3_strategy=['matched92','original_strategy'],Full_mapping='NEEDS_MAPPING')
 cfg['a3_patients']=sorted({r['patient'] for r in rows(O/'manifests/fixed_samples.csv') if r['cohort']=='validation'})
 js(W/'configs/analysis.json',cfg)
 files=set();source=[]
 for sub in ['scripts','configs','results','reports','manifests']:
  for p in (O/sub).rglob('*'):
   if p.is_file() and '__pycache__' not in str(p):files.add(str(p))
 for p in [O/'FINAL_REPORT_A2_A5_CN.md',O/'A2_A5_offline_evidence_20261007.tar.gz',ROOT/'revision_runs_20261006/state/queue.json',ROOT/'revision_runs_20261006/reports/source_index.csv',ROOT/'revision_runs_20261006/reports/frozen_source_versions.json']:
  files.add(str(p))
 for m in cfg['methods']:
  for cp in m['checkpoints']:
   assert sha(cp['path'])==cp['sha256'];files.add(cp['path'])
  for p in Path(m['run']).glob('*'):
   if p.is_file() and p.suffix in ['.py','.json','.md','.log']:files.add(str(p))
 for root in [R1,R2,ROOT/'revision_runs_20261006']:
  for sub in ['scripts','configs','source/train_ACDC']:
   for p in (root/sub).rglob('*'):
    if p.is_file() and p.suffix in ['.py','.yaml','.json']:files.add(str(p))
 for p in rows(O/'manifests/input_volumes.csv'):pass
 for r in rows(O/'manifests/fixed_samples.csv'):files.add(r['source'])
 files.add(str(DATA/'val.list'));files.add(str(DATA/'train_slices.list'))
 for f in sorted(files):source.append(dict(path=f,sha256=sha(f)))
 js(W/'manifests/protected.json',{r['path']:r['sha256'] for r in source});csvout(W/'manifests/protected_before.csv',source)
 for name in ['source_manifest.csv','fixed_samples.csv','input_volumes.csv']:
  shutil.copy2(O/'manifests'/name,W/'manifests'/name)
 # Archive small original evidence; checkpoints and full medical arrays remain read-only on server.
 for m in cfg['methods']:
  dest=W/'manifests/source_evidence'/m['id'];dest.mkdir(parents=True,exist_ok=True)
  for p in Path(m['run']).glob('*'):
   if p.is_file() and p.suffix in ['.py','.json','.md'] and p.stat().st_size<2000000:shutil.copy2(p,dest/p.name)
 for p in [R1/'scripts/revision_runtime.py',R1/'scripts/control_runtime.py',R2/'scripts/control_runtime.py',ROOT/'revision_runs_20261006/reports/source_index.csv',ROOT/'revision_runs_20261006/reports/frozen_source_versions.json']:
  dest=W/'manifests/source_evidence'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 src={str(p.relative_to(W)):sha(p) for p in (W/'scripts').rglob('*.py') if p.name not in ['finish.py','pipeline.py','launch.py','delivery.py']}
 fingerprint=dict(config=sha(W/'configs/analysis.json'),sources=src,protected=source)
 iid=__import__('hashlib').sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest();js(W/'manifests/identity.json',dict(id=iid,**fingerprint))
 js(W/'configs/tasklist.json',[dict(stage=s,authorized=True,training_updates=0) for s in ['A2_MAPPING','EQUIVALENCE','TIMING','A3','REVIEW','ARCHIVE']])
 atomic(W/'PRE_ANALYSIS_CN.md','''# 事前分析方案\n\n仅离线，训练/优化器/EMA更新均为0，旧E1=3976保持暂停。读取已完成A2/A5报告及代码，未发现适用AGENTS.md。Full作者说明保留，但来源层面待真实入口/绑定/轨迹对应，限定查询已知来源索引和四运行；无法唯一对应则NEEDS_MAPPING。ACDC测试集未用于原论文候选选择（作者更正）。\n\nA2复用原CSV，不新增前向；固定205 GT-present及210 fixed-three切片，10患者先同有效切片内等权、帧等权；best/last与raw/LCC分开，全像素最大概率和前景等权间隔均对所有方法计算，实际/辅助区别保留。\n\n部署四真实best student1，两路径仅关闭诊断。预设logits exact优先，allclose atol1e-6 rtol1e-5且mask/体hash完全相同；全部210切片、20体、4方法。每次加载预热50，5轮各100，方法/路径轮换；零输入与val.list首帧z0独立；FP32、deterministic=True、benchmark=False、matmulTF32=False、cudnnTF32=True沿用旧后端。CUDA events，host边界排除监测启动/退出；逐体路径3轮，不含文件IO/GT评价。GPU外来PID保留invalid重测。\n\nA3冻结原WeakStrongAugment与随机状态4261007，所有210验证切片按固定散列排序形成9批24个角色（最后6次循环暴露），12标注模拟角色+12无标注模拟角色，各子组6。GT不进入信号或权重。Full/R1/R2 best/last分别加载两学生及EMA。固定92匹配掩码位置由事前批次索引给定；R1/R2额外原策略以冻结教师batch信号生成矩形，位置按事前U(0,1)映射原upper-exclusive整数范围；Full原策略NA直到作者映射解决。\n\n原强增强cutout标签0和旋转填充无效区域独立记录，坐标适配器同RNG逐元素核验；记录共享几何及强颜色参数。四邻域多类边界两侧包含；主带宽欧氏距离<=2网格像素，1/3敏感性。四互斥区域使用未清理B_anat与seam，clean排除遮挡/几何无效及人工边界带，不重新归一化w。分位点0/.2/.4/.6/.8/1并列合并；空支持NA。uniform每图原均值，相同预测/误差，离线区域贡献对照。\n\n原始像素pooled与来源患者等权分别汇总；跨患者拼接按像素来源分配患者，每位10个真实患者而非重复配对独立样本。eval为唯一主模式，不执行train_snapshot。图预选patient002/019及clean边界富集最大/最近中位/最小。完整医学数组留服务器，仅匿名图和少量复核缓存入包。\n\n预计GPU推理/测速约10–20分钟，A3统计及图表约10–30分钟；代码核验和交付另约20–40分钟。GPU忙时等待，实际耗时日志为准。流水线独立tmux，单元原子完成，安全停止后只补缺项；重启需resume。五轮审核和实际解包SHA核验后交付。\n''')
 js(W/'reports/ENVIRONMENT.json',dict(time=now(),python=platform.python_version(),host=platform.node(),gpu=gpu(),pip=subprocess.check_output([PY,'-m','pip','freeze'],text=True),scope='no dependency installation'))
 print('PREPARED',iid,flush=True)
if __name__=='__main__':main()
