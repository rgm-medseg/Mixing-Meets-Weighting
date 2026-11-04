"""Editable vector table exports accompanying the two primary diagnostic figures."""
import numpy as np
from common import *
from figures import style,export,readcsv
import matplotlib.pyplot as plt
def tab(ax,data,columns,widths,title):
    ax.axis('off');ax.set_title(title,loc='left',fontweight='bold',fontsize=7,pad=10)
    t=ax.table(cellText=data,colLabels=columns,colWidths=widths,loc='center',cellLoc='right',colLoc='center',bbox=[0,0,1,.92]);t.auto_set_font_size(False);t.set_fontsize(6)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#b8b8b8');cell.set_linewidth(.35)
        if i==0:cell.set_facecolor('#eaf0f4');cell.set_text_props(weight='bold',fontsize=6)
        elif j==0:cell.set_text_props(ha='left')
    return t
def export_table(fig,slug,sources,description):
    export(fig,slug,sources,description)
    p=W/'figures'/f'{slug}_provenance.json';meta=json.loads(p.read_text())
    meta.update(raster_native_grid=None,embedded_rasters=False,case_selection='all specified cohort observations aggregated; no case selection',transforms='native vector table text from actual CSV; declared display rounding only')
    js(p,meta)
def main():
    with style():
        fig=plt.figure(figsize=(190/25.4,145/25.4));gs=fig.add_gridspec(2,1,left=.035,right=.98,top=.92,bottom=.14,hspace=.35)
        rr=readcsv('region_statistics_table.csv');data=[]
        for r in rr:data.append([r['region'].replace('_',' '),f"{int(r['pixels']):,}",f"{float(r['mean_e']):.5f}",f"{float(r['mean_w_CTW']):.3f}",f"{float(r['mean_we_CTW']):.5f}",f"{float(r['mean_gradient_uniform'])*1e9:.4f}",f"{float(r['mean_gradient_CTW'])*1e9:.4f}",f"{float(r['gradient_share_uniform_percent']):.2f}",f"{float(r['gradient_share_CTW_percent']):.2f}"])
        tab(fig.add_subplot(gs[0]),data,['Region','Pixels','Mean e','CTW w','CTW w×e','G(U)','G(CTW)','U share (%)','C share (%)'],[.16,.12,.1,.08,.11,.1,.1,.115,.115],'A  Joint consistency diagnostic: 108 mixed outputs per condition')
        rt=readcsv('trc_region_summary.csv');lookup={r['region']:r for r in rt};data=[]
        for name in ['RV','MYO','LV','anatomy','organ_interior','background','all']:
            r=lookup[name];data.append([name.replace('_',' '),f"{int(r['pixels']):,}",r['changed'],r['corrected'],r['introduced'],r['bothwrong_changed']])
        tab(fig.add_subplot(gs[1]),data,['Region','Pixels','Changed','Corrected','Introduced','Both wrong'],[.2,.2,.15,.15,.15,.15],'B  TRC intervention: 210 unmixed validation slices')
        fig.supxlabel('A: means and shares are patient-equal; pixel supports are pooled. G = L1 logit gradient × 10⁹ of the 0.05-scaled item.\nB: pooled counts for γ=0 → normal; anatomy / interior / background partition all pixels; organ rows use GT support.\nA and B use different inputs and denominators. Pseudo-label rows overlap the primary spatial partition.',fontsize=6)
        export_table(fig,'Table_regions',[str(W/'results/TABLE_regions.csv')],'Two panels: actual joint-loss regional statistics and TRC pixel changes; full patient and class statistics in CSV.')
        fixed=readcsv('fixed_control_table.csv');data=[]
        for r in fixed:
            data.append(['Original Full' if r['status']=='EXISTING_RETAINED' else 'Fixed 97×97',r['selected_step'],f"{float(r['validation_dice_percent']):.4f}",*[f"{float(r[k]):.4f}" for k in ['RV_dice_percent','MYO_dice_percent','LV_dice_percent','test_dice_percent','hd95_grid','asd_grid']]])
        fig,ax=plt.subplots(figsize=(190/25.4,58/25.4));fig.subplots_adjust(left=.03,right=.985,top=.86,bottom=.29)
        tab(ax,data,['Setting','Step','Val Dice (%)','RV (%)','MYO (%)','LV (%)','Test Dice (%)','HD95 (grid)','ASD (grid)'],[.17,.07,.13,.09,.1,.09,.13,.11,.11],'Fixed-region control: shared training/evaluation protocol, seed 42')
        fig.supxlabel('Each student: 30,000 SSL updates. Student 1: original validation-best selection. TRC and CTW on; original mean-max signal.\n97×97 matches 150 recorded main-Full endpoints, not a proven full-trajectory mean. Original Full test results are retained.',fontsize=6)
        export_table(fig,'Table_fixed_control',[str(W/'results/fixed_control_table.csv')],'Original Full and new fixed97 results, validation-selected checkpoints, classes and grid distances. Single-seed descriptive comparison.')
if __name__=='__main__':main()
