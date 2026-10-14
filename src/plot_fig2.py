"""Fig. 2: measured peeling responses, with original configured plateau means.
Read-only delivered inputs; no refitting or source writes.
Run through run_analysis.py with the installed requirements.
"""
from pathlib import Path
import os, argparse
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from package_paths import ROOT, RAW, CONFIG, PROCESSED, OUTPUT
from text_style import TEXT_RC, formatted
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from figure_data_compat import (apply_peeling_configuration, figure2_label,
                                normalize_peeling_config)

SOURCE=ROOT
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
parser.add_argument('--ethanol-time-thickness',type=int,choices=(50,100),default=int(json.loads((CONFIG/'figure_selection.json').read_text())['figure2']['ethanol_time_thickness_um']),
                    help='Thickness of the ethanol 3/6/9-h comparison in panel (a).')
args=parser.parse_args()
FIG=args.output_dir.resolve();DATA=FIG/'data'
for p in [FIG,DATA]:p.mkdir(parents=True,exist_ok=True)
from paper_style import (setup_publication_style, apply_axis_style, add_panel_label,
                         save_publication_figure, LIQUID_COLORS, LIQUID_MARKERS,
                         THICKNESS_COLORS, THICKNESS_MARKERS, TIME_LINES, TIME_MARKERS,
                         THICKNESS_LINES)
setup_publication_style()
# Display settings are local to Figure 2. Scientific inputs remain read-only.
from figure_settings import RAW_LINEWIDTH, RESPONSE_LINEWIDTH, THICKNESS_COLORS, LIQUID_COLORS
LABELS={'ethanol':'Ethanol','ipa':'IPA','water':'Water','glycerol':'Glycerol','oil':'Glycerol'}
config=normalize_peeling_config(pd.read_excel(CONFIG/'peeling_config.xlsx',sheet_name='peeling_config'))
results=pd.read_csv(PROCESSED/'peeling_results.csv')
results=results.loc[:,[c for c in results if not c.startswith('Unnamed')]]
results=apply_peeling_configuration(results, config)
# Conditions were selected by the author after comparison of all configured records.
# The quality audit remains descriptive; no trial or mechanical estimate is excluded.
# Existing windows and stored means remain unchanged.
ethanol_time_sheets=[]
for time in (3,6,9):
    matches=config[config.liquid.eq('ethanol') & config.thickness_um.eq(args.ethanol_time_thickness)
                   & config.drying_time_h.eq(time)]
    assert len(matches)==1,(args.ethanol_time_thickness,time)
    ethanol_time_sheets.append(matches.iloc[0].sheet_name)
figure_choices=json.loads((CONFIG/'figure_selection.json').read_text())['figure2']
selections={
 'a':{'sheets':ethanol_time_sheets,'title':'Force response',
      'condition':rf'Ethanol, $h={args.ethanol_time_thickness}\,\mu$m',
      'labels':[figure2_label(config,s,'time_speed') for s in ethanol_time_sheets],
      'xlim':(0,60),'ylim':(-3.8,37),'xticks':[0,20,40,60]},
 'b':{'sheets':figure_choices['liquid_series'],'title':'Liquid comparison',
      'condition':r'$h=50\,\mu$m; $t=3$ h',
      'labels':[figure2_label(config,s,'liquid_time') for s in figure_choices['liquid_series']],
      'xlim':(0,34),'ylim':(-4,24),'xticks':[0,10,20,30]},
 'c':{'sheets':figure_choices['ipa_thickness_series'],'title':'Thickness series',
      'condition':r'IPA, $t=3$ h',
      'labels':[r'25 $\mu$m',r'50 $\mu$m',r'100 $\mu$m',r'150 $\mu$m'],
      'xlim':(0,44),'ylim':(-3.2,36),'xticks':[0,10,20,30,40]},
}

raws={}; records=[]; metadata=[]
raw_book=pd.ExcelFile(RAW/'peeling/peeling_data.xlsx')
for panel,s in selections.items():
    for sheet,label in zip(s['sheets'],s['labels']):
        c=config.loc[config.sheet_name.eq(sheet)].iloc[0]
        r=results.loc[results.sheet_name.eq(sheet)].iloc[0]
        if sheet not in raws:
            raw=pd.read_excel(raw_book,sheet_name=sheet)
            raw=raw[['Time (s)','Load (N)','Displacement (mm)']].apply(pd.to_numeric,errors='coerce').dropna().reset_index(drop=True)
            raw['displacement_zeroed_mm']=raw['Displacement (mm)']-raw['Displacement (mm)'].iloc[0]
            raw['in_configured_plateau']=raw.displacement_zeroed_mm.between(float(c.plateau_start_mm_zeroed),float(c.plateau_end_mm_zeroed),inclusive='both')
            computed=raw.loc[raw.in_configured_plateau,'Load (N)'].mean()
            assert np.isclose(computed,float(r.plateau_force_mean_n),rtol=1e-11,atol=1e-13),(sheet,computed,r.plateau_force_mean_n)
            assert np.isclose(float(r.gamma_n_per_m),float(r.plateau_force_mean_n)/(float(c.specimen_width_mm)/1000),rtol=1e-11)
            raws[sheet]=raw
        raw=raws[sheet].copy()
        raw.insert(0,'panel',panel);raw.insert(1,'sheet_name',sheet)
        for key in ['liquid','drying_time_h','planned_drying_time_h','thickness_um','peel_angle_deg','peel_speed_mm_per_min','plateau_start_mm_zeroed','plateau_end_mm_zeroed']:
            raw[key]=c[key]
        raw['display_liquid']=LABELS[c.liquid]
        raw['stored_plateau_force_mean_n']=float(r.plateau_force_mean_n)
        raw['stored_gamma_J_m2']=float(r.gamma_n_per_m)
        records.append(raw)
        entry={'panel':panel,'sheet_name':sheet,'display_label':label,'display_liquid':LABELS[c.liquid]}
        for key in config:
            val=c[key]
            entry[key]=None if pd.isna(val) else val.item() if hasattr(val,'item') else val
        entry.update({'plateau_force_mean_n':float(r.plateau_force_mean_n),'gamma_J_m2':float(r.gamma_n_per_m),'raw_point_count':len(raws[sheet]),'plateau_point_count':int(raws[sheet].in_configured_plateau.sum()),
                      'source_raw':'data/peeling/peeling_data.xlsx','source_raw_sheet':sheet})
        metadata.append(entry)

raw_book.close()
binned={}
binned_records=[]
for sheet,d in raws.items():
    z=d.copy();z['bin_index']=np.floor(z.displacement_zeroed_mm/0.10).astype(int)
    v=z.groupby('bin_index',as_index=False).agg(displacement_mm=('displacement_zeroed_mm','mean'),force_n=('Load (N)','mean'),raw_points=('Load (N)','size'))
    v['bin_left_mm']=v.bin_index*0.10;v['bin_right_mm']=v.bin_left_mm+0.10;v['sheet_name']=sheet
    binned[sheet]=v;binned_records.append(v)
pd.concat(binned_records,ignore_index=True).to_csv(DATA/'fig2_display_bin_means.csv',index=False)

pd.concat(records,ignore_index=True).to_csv(DATA/'fig2_selected_curves.csv',index=False)
pd.DataFrame(metadata).to_csv(DATA/'fig2_selected_specimens.csv',index=False)


def draw(ax,panel,full=False):
    s=selections[panel]
    time_colors=dict(zip((3,6,9),(THICKNESS_COLORS[25],THICKNESS_COLORS[50],THICKNESS_COLORS[100])))
    for i,(sheet,label) in enumerate(zip(s['sheets'],s['labels'])):
        r=results.loc[results.sheet_name.eq(sheet)].iloc[0];d=raws[sheet]
        c=config.loc[config.sheet_name.eq(sheet)].iloc[0]
        liquid=LABELS[c.liquid];h=int(c.thickness_um);time=int(c.drying_time_h)
        color=time_colors[time] if panel=='a' else LIQUID_COLORS[liquid] if panel=='b' else THICKNESS_COLORS[h]
        marker=TIME_MARKERS[time] if panel=='a' else LIQUID_MARKERS[liquid] if panel=='b' else THICKNESS_MARKERS[h]
        line=TIME_LINES[time] if panel=='a' else '-' if panel=='b' else THICKNESS_LINES[h]
        if not full:
            d=d[d.displacement_zeroed_mm.le(float(r.plateau_end_mm_zeroed))]
        # Raw points are shown faintly. Fixed-width bin means are display-only.
        ax.plot(d.displacement_zeroed_mm,d['Load (N)']*1000,color=color,lw=RAW_LINEWIDTH,alpha=.20,zorder=2)
        v=binned[sheet]
        if not full:
            v=v[v.displacement_mm.le(float(r.plateau_end_mm_zeroed))]
        ax.plot(v.displacement_mm,v.force_n*1000,color=color,lw=RESPONSE_LINEWIDTH,alpha=1,ls=line,marker=marker,markevery=40,markersize=2.6,markerfacecolor='white',markeredgewidth=.7,zorder=3)
        start=float(r.plateau_start_mm_zeroed);end=float(r.plateau_end_mm_zeroed);mean=float(r.plateau_force_mean_n)*1000
        ax.plot([start,end],[mean,mean],color='white',lw=3.6,zorder=9)
        ax.plot([start,end],[mean,mean],color='black',lw=1.55,ls=(0,(4,2)),zorder=10)
        ax.plot([start,end],[mean,mean],color='black',marker='|',markersize=8,mew=1.2,lw=0,zorder=11)
    ax.set_xlim(s['xlim'] if not full else (0,max(float(raws[sh].displacement_zeroed_mm.max()) for sh in s['sheets'])*1.025))
    ax.set_ylim(s['ylim'])
    if full:
        visible=[raws[sh].loc[raws[sh].displacement_zeroed_mm.between(*ax.get_xlim()),'Load (N)'].to_numpy()*1000 for sh in s['sheets']]
        all_y=np.concatenate(visible);lo=float(np.min(all_y));hi=float(np.max(all_y));pad=(hi-lo)*.08
        ax.set_ylim(lo-pad,hi+(hi-lo)*(.50 if panel=='a' else .24))
    apply_axis_style(ax)
    ax.set_xlabel(r'Displacement, $\delta$ (mm)')
    ax.set_ylabel(r'Peel force, $P$ (mN)',labelpad=4)
    add_panel_label(ax,panel,s['title'],s['condition'])
    handles=[]
    for sheet,label in zip(s['sheets'],s['labels']):
        c=config.loc[config.sheet_name.eq(sheet)].iloc[0]
        liquid=LABELS[c.liquid];h=int(c.thickness_um);time=int(c.drying_time_h)
        color=time_colors[time] if panel=='a' else LIQUID_COLORS[liquid] if panel=='b' else THICKNESS_COLORS[h]
        marker=TIME_MARKERS[time] if panel=='a' else LIQUID_MARKERS[liquid] if panel=='b' else THICKNESS_MARKERS[h]
        line=TIME_LINES[time] if panel=='a' else '-' if panel=='b' else THICKNESS_LINES[h]
        display_label=label.replace('mm min$^{-1}$','mm/min')
        handles.append(Line2D([],[],color=color,lw=RESPONSE_LINEWIDTH,ls=line,marker=marker,markersize=3,mfc='white',label=display_label))
    ax.legend(handles=handles,loc='upper right',handlelength=1.45,labelspacing=.3,
              borderaxespad=.35,ncol=1 if panel=='a' else 2,columnspacing=.6,
              fontsize=6.5,title='Time / configured speed' if panel=='a' else None,title_fontsize=6.5)



def draw_manuscript(ax,panel):
    """Retain the manuscript layout, with a white, unshaded plotting area.

    The original configured windows still define the horizontal mean segments.
    Only their background fill is omitted; raw data and averaging are unchanged.
    """
    s=selections[panel]
    time_colors=dict(zip((3,6,9),(THICKNESS_COLORS[25],THICKNESS_COLORS[50],THICKNESS_COLORS[100])))
    handles=[]
    for sheet,label in zip(s['sheets'],s['labels']):
        r=results.loc[results.sheet_name.eq(sheet)].iloc[0]
        c=config.loc[config.sheet_name.eq(sheet)].iloc[0]
        liquid=LABELS[c.liquid];h=int(c.thickness_um);time=int(c.drying_time_h)
        color=time_colors[time] if panel=='a' else LIQUID_COLORS[liquid] if panel=='b' else THICKNESS_COLORS[h]
        marker=TIME_MARKERS[time] if panel=='a' else LIQUID_MARKERS[liquid] if panel=='b' else THICKNESS_MARKERS[h]
        line=TIME_LINES[time] if panel=='a' else '-' if panel=='b' else THICKNESS_LINES[h]
        start=float(r.plateau_start_mm_zeroed);end=float(r.plateau_end_mm_zeroed)
        mean=float(r.plateau_force_mean_n)*1000
        d=raws[sheet].loc[raws[sheet].displacement_zeroed_mm.le(end)]
        v=binned[sheet].loc[binned[sheet].displacement_mm.le(end)]
        ax.plot(d.displacement_zeroed_mm,d['Load (N)']*1000,color=color,lw=RAW_LINEWIDTH,alpha=.20,zorder=2)
        ax.plot(v.displacement_mm,v.force_n*1000,color=color,lw=RESPONSE_LINEWIDTH,ls=line,
                marker=marker,markevery=40,markersize=2.6,markerfacecolor='white',markeredgewidth=.7,zorder=3)
        # Mean segments identify the averaging windows without background spans.
        ax.plot([start,end],[mean,mean],color='black',lw=1.1,alpha=1,zorder=10)
        display_label=f'{time} h' if panel=='a' else liquid if panel=='b' else label
        handles.append(Line2D([],[],color=color,lw=RESPONSE_LINEWIDTH,ls=line,marker=marker,
                              markersize=3,mfc='white',label=display_label))
    ax.set(xlim=s['xlim'],ylim=s['ylim'],facecolor='white')
    ax.set_xticks(s['xticks'])
    apply_axis_style(ax)
    ax.spines[['top','right']].set_visible(True)
    ax.set_xlabel(r'$\delta$ (mm)')
    ax.set_ylabel(r'Peeling force, $P$ (mN)',labelpad=4)
    layout=json.loads((CONFIG/'figure_layout.json').read_text(encoding='utf-8'))['figure2']
    ax.annotate('('+panel+')',xy=(0,1),xycoords='axes fraction',
                xytext=layout['panel_offset_pt'],textcoords='offset points',fontsize=12,
                va='bottom',annotation_clip=False)
    condition={'a':rf'Ethanol, {args.ethanol_time_thickness} $\mu$m','b':r'50 $\mu$m, 3 h','c':'IPA, 3 h'}[panel]
    location={'a':'lower right','b':'upper right','c':'lower right'}[panel]
    legend=ax.legend(handles=handles,loc=location,title=condition,title_fontsize=9.5,
              handlelength=1.2,labelspacing=.25,borderaxespad=.3,ncol=1,
              fontsize=8.5,handletextpad=.35,borderpad=.35,
              frameon=True,fancybox=False,framealpha=1,facecolor='white',edgecolor='black')
    legend.get_frame().set_linewidth(.8)


# The manuscript figure used TeX serif text and a boxed three-panel layout.
# Keep that presentation reproducible here instead of editing exported images.
with plt.rc_context({**TEXT_RC,'axes.spines.top':True,'axes.spines.right':True}):
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.1))
    fig.subplots_adjust(left=.065,right=.98,bottom=.19,top=.85,wspace=.35)
    for ax,p in zip(axes,'abc'):draw_manuscript(ax,p)
    save_publication_figure(fig,FIG/'Fig2_peeling')
    plt.close(fig)
print('Figure 2 generated in', FIG)
