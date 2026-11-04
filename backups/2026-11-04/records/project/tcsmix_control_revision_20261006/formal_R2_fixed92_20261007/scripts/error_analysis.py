"""CPU-only post-test geometry audit of frozen predictions; never changes any output mask."""
import argparse,collections,inspect
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from medpy.metric import binary
from common import *
from unified_evaluation import metrics

TARGETS={'patient065','patient075','patient083'}
CLASSES={1:'RV',2:'MYO',3:'LV'}
KEYS=['dice','iou','hd95','asd']
SIX=ndi.generate_binary_structure(3,1)

def span(mask):
    idx=np.flatnonzero(mask.any((1,2)))
    return ('NA','NA') if not len(idx) else (int(idx[0]),int(idx[-1]))

def surface(mask):return mask ^ ndi.binary_erosion(mask,structure=SIX,iterations=1,border_value=0)

def geometry(pred,gt,model,case,c,full):
    row=dict(model=model,case=case,patient=case.split('_')[0],class_id=c,class_name=CLASSES[c],unit='grid',**metrics(pred,gt))
    tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum())
    row.update(TP=tp,FP=fp,FN=fn,pred_pixels=int(pred.sum()),GT_pixels=int(gt.sum()),pred_first_slice=span(pred)[0],pred_last_slice=span(pred)[1],GT_first_slice=span(gt)[0],GT_last_slice=span(gt)[1])
    zz=gt.any((1,2));row['pred_pixels_on_GT_empty_slices']=int(pred[~zz].sum())
    slices=[];components=[];distrows=[];distbins=[];points=[];selection={}
    for z in range(len(pred)):
        p,g=pred[z],gt[z];t=int((p&g).sum());f=int((p&~g).sum());n=int((~p&g).sum());den=int(p.sum()+g.sum());union=int((p|g).sum())
        slices.append(dict(model=model,case=case,class_name=CLASSES[c],z=z,TP=t,FP=f,FN=n,pred_area=int(p.sum()),GT_area=int(g.sum()),dice=2*t/den if den else 0.,iou=t/union if union else 0.))
    if not full:return row,slices,components,distrows,distbins,points,selection
    labels,ncomp=ndi.label(pred,structure=SIX);gl,ng=ndi.label(gt,structure=SIX)
    sizes=np.bincount(labels.ravel())[1:];gs=np.bincount(gl.ravel())[1:]
    row.update(pred_components_6=ncomp,GT_components_6=ng,pred_largest_component_fraction=float(sizes.max()/sizes.sum()) if sizes.size else 0.,GT_largest_component_fraction=float(gs.max()/gs.sum()) if gs.size else 0.)
    dist_gt=ndi.distance_transform_edt(~gt) if gt.any() else None
    for component,sl in enumerate(ndi.find_objects(labels),1):
        if sl is None:continue
        local=labels[sl]==component;coords=np.argwhere(local)+np.array([s.start for s in sl]);index=tuple(coords.T)
        d=dist_gt[index] if dist_gt is not None else None
        components.append(dict(model=model,case=case,class_name=CLASSES[c],component=component,connectivity='6-neighbor',pixels=len(coords),fraction=len(coords)/max(1,int(pred.sum())),GT_overlap=int(gt[index].sum()),z_first=int(coords[:,0].min()),z_last=int(coords[:,0].max()),
            min_distance_to_GT_voxels=float(d.min()) if d is not None else 'NA',mean_distance_to_GT_voxels=float(d.mean()) if d is not None else 'NA',max_distance_to_GT_voxels=float(d.max()) if d is not None else 'NA',FP_distance_ge10=int((d>=10).sum()) if d is not None else 'NA',FP_distance_ge20=int((d>=20).sum()) if d is not None else 'NA',threshold_role='description only, no prediction postprocess'))
    row['zero_GT_overlap_components']=sum(x['GT_overlap']==0 for x in components)
    row['remote_zero_overlap_components_ge10']=sum(x['GT_overlap']==0 and x['min_distance_to_GT_voxels']!='NA' and x['min_distance_to_GT_voxels']>=10 for x in components)
    if not pred.any() or not gt.any():
        row['surface_scope']='empty-mask original metric policy; no nearest surface distances'
        return row,slices,components,distrows,distbins,points,selection
    ps,gs=surface(pred),surface(gt);pc=np.argwhere(ps);gc=np.argwhere(gs)
    dt,nearest_gt=ndi.distance_transform_edt(~gs,return_indices=True)
    dp=dt[ps];matchgt=nearest_gt[:,ps].T
    del dt,nearest_gt
    dt,nearest_pred=ndi.distance_transform_edt(~ps,return_indices=True)
    dg=dt[gs];matchpred=nearest_pred[:,gs].T
    del dt,nearest_pred
    assert abs(float(dp.mean())-row['asd'])<1e-10
    assert abs(float(np.percentile(np.r_[dp,dg],95))-row['hd95'])<1e-10
    row.update(pred_surface_points=len(dp),GT_surface_points=len(dg),surface_scope='6-neighbor erosion borders; Euclidean unit grid; HD95 concatenates both directions; ASD pred-to-GT only')
    surface_component=labels[ps]
    remote_ids=[x['component'] for x in components if x['GT_overlap']==0 and x['min_distance_to_GT_voxels']>=10]
    unmatched_ids=[x['component'] for x in components if x['GT_overlap']==0]
    remote=np.isin(surface_component,remote_ids);unmatched=np.isin(surface_component,unmatched_ids)
    row['remote_component_ASD_contribution']=float(dp[remote].sum()/len(dp))
    row['zero_overlap_component_ASD_contribution']=float(dp[unmatched].sum()/len(dp))
    row['remote_component_fraction_of_pred_distance_sum']=float(dp[remote].sum()/dp.sum()) if dp.sum() else 0.
    row['remote_component_HD95_pred_tail_points']=int((remote&(dp>=row['hd95'])).sum())
    for component in components:
        values=dp[surface_component==component['component']]
        component.update(pred_surface_points=len(values),directed_ASD_contribution=float(values.sum()/len(dp)),
            surface_distance_mean=float(values.mean()) if len(values) else 'NA',surface_distance_max=float(values.max()) if len(values) else 'NA',HD95_pred_tail_points=int((values>=row['hd95']).sum()))
    directional={}
    for direction,ds,coords,nearest in [('pred_to_GT',dp,pc,matchgt),('GT_to_pred',dg,gc,matchpred)]:
        summary=dict(model=model,case=case,class_name=CLASSES[c],direction=direction,z='ALL',points=len(ds),distance_sum=float(ds.sum()),mean=float(ds.mean()),maximum=float(ds.max()),tail_ge_volume_HD95=int((ds>=row['hd95']).sum()),distance_ge10=int((ds>=10).sum()),distance_ge20=int((ds>=20).sum()))
        summary.update({f'p{k}':float(np.percentile(ds,k)) for k in [0,50,75,90,95,99,100]});distrows.append(summary)
        values,counts=np.unique(ds,return_counts=True)
        distbins.extend(dict(model=model,case=case,class_name=CLASSES[c],direction=direction,distance=float(value),count=int(count),unit='grid') for value,count in zip(values,counts))
        perz=[]
        for z in range(len(pred)):
            values=ds[coords[:,0]==z];rr=dict(model=model,case=case,class_name=CLASSES[c],direction=direction,z=z,points=len(values),distance_sum=float(values.sum()),mean=float(values.mean()) if len(values) else 'NA',maximum=float(values.max()) if len(values) else 'NA',p95=float(np.percentile(values,95)) if len(values) else 'NA',tail_ge_volume_HD95=int((values>=row['hd95']).sum()),tail_distance_sum=float(values[values>=row['hd95']].sum()),distance_ge10=int((values>=10).sum()),distance_ge20=int((values>=20).sum()))
            if direction=='pred_to_GT':rr['contribution_to_directed_volume_ASD']=float(values.sum()/len(dp))
            distrows.append(rr);perz.append(rr)
        directional[direction]=perz
        top=np.argsort(-ds,kind='stable')[:20]
        for rank,i in enumerate(top,1):
            z,y,x=map(int,coords[i]);nz,ny,nx=map(int,nearest[i])
            points.append(dict(model=model,case=case,class_name=CLASSES[c],direction=direction,rank=rank,distance=float(ds[i]),z=z,y=y,x=x,nearest_z=nz,nearest_y=ny,nearest_x=nx,source_voxel_pred=bool(pred[z,y,x]),source_voxel_GT=bool(gt[z,y,x]),pred_component=int(labels[z,y,x]),GT_empty_slice=not bool(zz[z]),unit='grid'))
    selection['largest_directed_ASD_slice_contribution']=max(directional['pred_to_GT'],key=lambda r:(r['distance_sum'],-r['z']))['z']
    selection['largest_HD95_tail_distance_sum']=max(range(len(pred)),key=lambda z:(sum(r[z]['tail_distance_sum'] for r in directional.values()),-z))
    # These features describe geometries; no unique causal attribution from two checkpoints.
    row['pred_surface_ge10_fraction']=float((dp>=10).mean());row['GT_surface_ge10_fraction']=float((dg>=10).mean())
    row['pred_tail_on_GT_empty_slices']=int(((dp>=row['hd95'])&~zz[pc[:,0]]).sum())
    row['GT_tail_on_pred_empty_slices']=int(((dg>=row['hd95'])&~pred.any((1,2))[gc[:,0]]).sum())
    return row,slices,components,distrows,distbins,points,selection

def main():
    p=argparse.ArgumentParser();p.add_argument('--include-r2',action='store_true');a=p.parse_args()
    label='R1_R2' if a.include_r2 else 'R1';out=WORK/'analysis'/('error_'+label);out.mkdir(exist_ok=True)
    receipt=out/'ERROR_ANALYSIS_RECEIPT.json'
    if receipt.exists() and read_json(receipt)['status']=='DONE':print('Reused completed frozen prediction geometry:',receipt);return
    methods={
        'Old_Full':dict(root=PROJECT/'revision_runs_20261006/runs/Historical_Full/evaluation',csv='case_class_metrics.csv',receipt='EVALUATION_RECEIPT.json'),
        'R1_v2_B':dict(root=PARENT_R1/'results',csv='case_class_metrics.csv',receipt='EVALUATION_RECEIPT.json')}
    if a.include_r2:methods['R2_fixed92']=dict(root=WORK/'results',csv='case_class_metrics.csv',receipt='EVALUATION_RECEIPT.json')
    identities={};expected={};missing=[]
    for method,cfg in methods.items():
        if not (cfg['root']/cfg['receipt']).exists():missing.append(dict(method=method,case='ALL',reason='evaluation receipt missing'));continue
        r=read_json(cfg['root']/cfg['receipt']);assert r['protocol']=='ACDC_v22_grid_directed_ASD'
        assert r.get('test_manifest_sha256',r.get('manifest_sha256'))==sha(DATA/'test.list')
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        identities[method]=r;expected[method]={(x['case'],x['class_name']):x for x in read_table(cfg['root']/cfg['csv'])}
    allrows=[];slices=[];comps=[];dists=[];bins=[];points=[];index=[];selections={};target_predictions={}
    for case in (DATA/'test.list').read_text().splitlines():
        src=DATA/'data'/f'{case}.h5'
        if not src.exists():missing.append(dict(method='ALL',case=case,reason='GT/image source missing'));continue
        with h5py.File(src,'r') as f:gt=f['label'][:]
        for method,cfg in methods.items():
            path=cfg['root']/'predictions'/f'{case}.npz'
            if method not in identities or not path.exists():missing.append(dict(method=method,case=case,reason='frozen prediction or receipt unavailable; no GPU inference scheduled during R2 training'));continue
            with np.load(path) as f:
                pred=f['prediction'];assert str(f['case'])==case
                if 'checkpoint_sha256' in f:assert str(f['checkpoint_sha256'])==identities[method]['checkpoint_sha256']
                else:assert str(f['checkpoint'])==identities[method]['checkpoint']
            assert pred.shape==gt.shape and set(np.unique(pred))<=set(range(4))
            targeted=case.split('_')[0] in TARGETS
            if targeted:target_predictions[(method,case)]=pred
            index.append(dict(method=method,case=case,prediction=str(path),prediction_sha256=sha(path),GT=str(src),GT_sha256=sha(src),checkpoint=identities[method]['checkpoint'],checkpoint_sha256=identities[method]['checkpoint_sha256'],shape='x'.join(map(str,pred.shape)),axes='z,y,x',spacing='1,1,1',unit='grid',postprocessing='none'))
            for c in [1,2,3]:
                row,ss,cc,dd,bb,pp,zz=geometry(pred==c,gt==c,method,case,c,targeted)
                prior=expected[method][(case,CLASSES[c])]
                assert all(abs(row[k]-float(prior[k]))<1e-9 for k in KEYS),(method,case,c,row,prior)
                allrows.append(row);slices.extend(ss);comps.extend(cc);dists.extend(dd);bins.extend(bb);points.extend(pp)
                if targeted:selections[(method,case,c)]=zz
    # Rectangular CSV: aggregate rows outside priority cases deliberately lack geometry details.
    fields=list(dict.fromkeys(k for r in allrows for k in r));write_csv(out/'mask_volume_metrics.csv',allrows,fields)
    write_csv(out/'slice_errors.csv',slices)
    write_csv(out/'connected_components.csv',comps)
    fields=list(dict.fromkeys(k for r in dists for k in r));write_csv(out/'surface_distance_statistics.csv',dists,fields)
    write_csv(out/'surface_distance_histogram.csv',bins);write_csv(out/'largest_surface_distance_points.csv',points);write_csv(out/'prediction_provenance.csv',index)
    if missing:write_csv(out/'missing_inputs.csv',missing)
    lookup={(r['model'],r['case'],r['class_name']):r for r in allrows};paired=[]
    for r in allrows:
        old=lookup.get(('Old_Full',r['case'],r['class_name']))
        if r['model']=='Old_Full' or old is None:continue
        paired.append(dict(model=r['model'],case=r['case'],patient=r['patient'],class_name=r['class_name'],**{k+'_delta':r[k]-old[k] for k in KEYS}))
    write_csv(out/'paired_case_class_metrics.csv',paired)
    patients=[];classes=[]
    for model in sorted({r['model'] for r in paired}):
        for patient in sorted({r['patient'] for r in paired if r['model']==model}):
            rr=[r for r in paired if r['model']==model and r['patient']==patient]
            patients.append(dict(model=model,patient=patient,n_rows=len(rr),**{k+'_delta':float(np.mean([r[k+'_delta'] for r in rr])) for k in KEYS}))
        for cls in CLASSES.values():
            rr=[r for r in paired if r['model']==model and r['class_name']==cls]
            classes.append(dict(model=model,class_name=cls,**{k+'_delta':float(np.mean([r[k+'_delta'] for r in rr])) for k in KEYS}))
    write_csv(out/'patient_deltas.csv',patients);write_csv(out/'class_deltas.csv',classes)
    r1paired=[r for r in paired if r['model']=='R1_v2_B'];r1patients=[r for r in patients if r['model']=='R1_v2_B']
    shares={}
    for key in ['hd95','asd']:
        net=sum(r[key+'_delta'] for r in r1paired);target=sum(r[key+'_delta'] for r in r1paired if r['patient'] in {'patient065','patient075'})
        shares[key]=dict(all_net_sum=net,two_patients_net_sum=target,net_share=target/net if net else 'NA',meaning='signed delta sum share, includes cancellation; not exclusive cause attribution or exclusion-adjusted main score')
    selected=[]
    for case in sorted({r['case'] for r in allrows if r['patient'] in TARGETS}):
        deltas=[r for r in r1paired if r['case']==case]
        if not deltas:continue
        worst=max(deltas,key=lambda r:(r['hd95_delta'],r['class_name']));c=next(c for c,n in CLASSES.items() if n==worst['class_name'])
        choices=selections.get(('R1_v2_B',case,c),{})
        source=DATA/'data'/f'{case}.h5'
        with h5py.File(source,'r') as f:image=f['image'][:];gt=f['label'][:]
        for rule,z in choices.items():
            if not all((method,case) in target_predictions for method in methods):continue
            fig,axes=plt.subplots(1,2+len(methods),figsize=(3*(2+len(methods)),3.3),layout='constrained')
            base=image[z];lo,hi=np.percentile(base,[1,99]);base=np.clip((base-lo)/max(hi-lo,1e-6),0,1)
            masks=[None,gt[z]]+[target_predictions[(m,case)][z] for m in methods]
            names=['Image','GT']+list(methods)
            for ax,name,mask in zip(axes,names,masks):
                ax.imshow(base,cmap='gray',vmin=0,vmax=1,origin='upper',interpolation='nearest')
                if mask is not None:
                    for cls,color in [(1,'#0072B2'),(2,'#E69F00'),(3,'#D55E00')]:
                        if (mask==cls).any():ax.contour(mask==cls,levels=[.5],colors=[color],linewidths=.8)
                ax.set_title(name);ax.axis('off')
            fig.suptitle(f'{case} | {CLASSES[c]} | z={z} | {rule}\nRV blue / MYO orange / LV red; native z,y,x grid, identical orientation & scale',fontsize=9)
            name=f'error_{case}_{CLASSES[c]}_{rule}'
            for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=160,bbox_inches='tight')
            plt.close(fig)
            for method in methods:
                volume=lookup[(method,case,CLASSES[c])];slice_row=next(r for r in slices if r['model']==method and r['case']==case and r['class_name']==CLASSES[c] and r['z']==z)
                selected.append(dict(rule='largest R1 minus old HD95 class within frame; '+rule,case=case,patient=case.split('_')[0],class_name=CLASSES[c],z=z,method=method,slice_dice=slice_row['dice'],slice_iou=slice_row['iou'],slice_FP=slice_row['FP'],slice_FN=slice_row['FN'],volume_dice=volume['dice'],volume_iou=volume['iou'],volume_hd95=volume['hd95'],volume_asd=volume['asd'],figure=str((out/f'{name}.png').relative_to(WORK)),axes='z,y,x; origin upper',deidentification='anonymous ACDC IDs, normalized image-only render, no DICOM headers or raw medical arrays'))
    write_csv(out/'qualitative_selection.csv',selected)
    # Evidence-based descriptions of geometry for every priority frame and class, both gains and losses.
    descriptions=[]
    for r in allrows:
        if r['model']!='R1_v2_B' or r['patient'] not in TARGETS:continue
        old=lookup.get(('Old_Full',r['case'],r['class_name']))
        if old is None:continue
        features=[]
        if r.get('remote_zero_overlap_components_ge10',0):features.append(f"{r['remote_zero_overlap_components_ge10']}个与GT零交叠且最小距离≥10 grid的孤立域")
        if r['pred_pixels_on_GT_empty_slices']:features.append(f"GT无该类切片上{r['pred_pixels_on_GT_empty_slices']}个预测像素（旧{old['pred_pixels_on_GT_empty_slices']}）")
        if r['FN']>old['FN']:features.append(f"FN增加{r['FN']-old['FN']}像素")
        if r['FP']>old['FP']:features.append(f"FP增加{r['FP']-old['FP']}像素")
        features.append(f"远端孤立域对单向ASD的加性贡献差{r.get('remote_component_ASD_contribution',0)-old.get('remote_component_ASD_contribution',0):+.3f} grid")
        descriptions.append(f"- {r['case']} {r['class_name']}：Dice差{100*(r['dice']-old['dice']):+.3f}pp，HD95差{r['hd95']-old['hd95']:+.3f}，单向ASD差{r['asd']-old['asd']:+.3f}；"+('；'.join(features) if features else '未检出上述新增远端/范围或计数特征，局部轮廓差异需结合表面点和对照图判断')+'。')
    status='PARTIAL' if missing or len(r1paired)!=120 else 'DONE'
    review=dict(status=status,time=now(),CPU_only=True,optimizer_updates=0,GPU_inference=0,models=list(methods),volume_class_rows=len(allrows),recomputed_metrics_match_frozen_CSV=True,
        R1_patients_Dice_declined=sum(r['dice_delta']<0 for r in r1patients),R1_patients_count=len(r1patients),R1_class_deltas=[r for r in classes if r['model']=='R1_v2_B'],two_patient_net_contribution=shares,
        connectivity='6-neighbor (scipy generate_binary_structure(3,1)); erosion border_value=0; unit spacing1',surface_definition='HD95 pooled bidirectional border distances; ASD pred-to-GT only',missing=missing,
        priority_cases=sorted(TARGETS),test_use='post-test error description only; no controller, area, checkpoint or postprocessing changes',R2_included=a.include_r2)
    atomic_json(receipt,review)
    (out/'MEDPY_SURFACE_DEFINITION.txt').write_text(inspect.getsource(binary.__surface_distances))
    report=f'''# R1主要误差病例核查（{'含R2最终几何对照' if a.include_r2 else 'R2启动前，已有预测CPU分析'}）

状态{status}，时刻{now()}。读取真实冻结检查点预测与GT，重新计算全部40个frame×3类指标并逐项匹配原CSV（误差<1e-9）；未训练、未修改预测/GT/后处理，GPU推理0次。R1选择step29000；旧Full选择step29600。来源、数组z,y,x、原尺寸、预测/GT/检查点哈希见{out.relative_to(WORK)}/prediction_provenance.csv。

20病例中R1 Dice下降{review['R1_patients_Dice_declined']}例；分类差值（fraction，×100为pp）：{review['R1_class_deltas']}。
patient065与075对HD95、ASD有符号净增量的贡献复核为：{shares}。这是全体差值中可含正负抵消的净占比，不能将两个病例当成全部原因；全体20病例、120类frame行均保留，不生成剔除病例主结果。

三维连通域与表面均用6邻域。HD95为预测→GT和GT→预测两组表面距离拼接后的95分位，ASD只用预测→GT。表面统计均在原H5 grid、spacing=(1,1,1)，不是mm。完整距离频数、逐slice距离贡献、最大20个表面点及其最近对应点坐标均另存CSV；复算均与原评价一致。诊断中≥10/20 grid仅描述远端程度，不是模型阈值或新增后处理。

确认的几何现象（各优先frame的全部类别）：
{chr(10).join(descriptions)}

表面点方向可区分远端假阳性与漏分割的位置；范围延伸用GT空切片预测像素描述，漏分割用FN/GT→预测尾部描述。孤立域、范围延伸和局部偏移可能共存，这些是两个冻结模型输出的现象，不证明某个训练模块造成错误。没有把所有距离变化统称为孤立假阳性。

定性选择：在每个优先frame内选HD95差最大的类别；分别选R1预测→GT距离总和最高的slice（其对单向ASD贡献最大），以及双向≥该体积HD95尾部距离总和最高的slice；逐规则首次tie。不是最大GT面积slice。统一原生方向、颜色、图像强度范围及尺度，图中所有类均显示，具体slice/class/volume指标见qualitative_selection.csv。匿名ACDC ID，不携带DICOM元数据；原始医学数组/全预测不打包。

证据缺口：{missing or '上述预测、GT和评价均可用；这里只覆盖现有冻结输出，不是全程机制因果或跨种子证据'}。R1 q_B与质量使用此前固定标注/验证样本的独立诊断结果，另存analysis/R1_confidence_quality*.csv，与此测试误差描述分开。固定92已按全程面积锁定，未按这些误差调整。
'''
    target=WORK/('R1_R2_ERROR_ANALYSIS_CN.md' if a.include_r2 else 'R1_ERROR_ANALYSIS_CN.md');target.write_text(report)
    print(json.dumps(review,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
