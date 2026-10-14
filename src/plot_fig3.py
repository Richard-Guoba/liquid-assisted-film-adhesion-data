"""Figure 3 uses freshly fitted force/radius values and original video frames."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from package_paths import ROOT,RAW,CONFIG,PROCESSED,OUTPUT
from figure_settings import THICKNESS_COLORS,FONT_FAMILY
from paper_style import setup_publication_style,save_publication_figure
from photo_frames import approved_photos
from text_style import TEXT_RC

parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=OUTPUT);args=parser.parse_args()
OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True);DATA=OUT/'data';DATA.mkdir(exist_ok=True)
selection=json.loads((CONFIG/'figure_selection.json').read_text(encoding='utf-8'))['figure3']
layout=json.loads((CONFIG/'figure_layout.json').read_text(encoding='utf-8'))['figure3']
force=pd.read_csv(PROCESSED/'force_fit_results.csv');radius=pd.read_csv(PROCESSED/'common_points.csv')
common=pd.read_csv(PROCESSED/'radius_common.csv')
rep=force[force.test_id.eq(selection['representative_force_id'])].iloc[0]
rp=radius[radius['sample'].eq(selection['representative_radius_sample'])].sort_values('point_order')
rc=common[common.specimen_id.eq(rep.test_id)].iloc[0]
book=pd.ExcelFile(RAW/'force/force_data.xlsx');raw=book.parse(rep.test_id)
setup_publication_style();plt.rcParams.update({'font.family':FONT_FAMILY,'mathtext.fontset':'stix','font.size':9.5,
    'axes.labelsize':10.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'axes.linewidth':.9,**TEXT_RC})
fig=plt.figure(figsize=(7.4,5.45),facecolor='white')
axf=fig.add_axes([.085,.72,.325,.22]);axa=fig.add_axes([.085,.395,.325,.22]);axm=fig.add_axes([.565,.395,.42,.545])
for ax,letter in [(axf,'a'),(axa,'b'),(axm,'d')]:
    ax.tick_params(direction='in',top=True,right=True,length=3.5,width=.8,pad=4)
    ax.annotate(f'({letter})',xy=(0,1),xycoords='axes fraction',xytext=layout['panel_offsets_pt'][letter],textcoords='offset points',fontsize=12,va='bottom',annotation_clip=False)
fit_x=np.array([rep.fit_curve_region_start_mm,rep.fit_curve_region_end_mm])
axf.plot(raw['Displacement (mm)'],raw['Load (N)'],color='red',lw=.75,label='Measured')
axf.plot(fit_x,rep.slope_n_per_mm*fit_x+rep.fit_intercept_n,color='black',ls=(0,(4.5/2.05,3/2.05)),lw=2.05,label='Linear fit')
axf.set(xlim=(0,10.5),ylim=(-.025,.86),xlabel=r'$\delta$ (mm)',ylabel=r'$F$ (N)',xticks=[0,5,10],yticks=[0,.4,.8])
axf.legend(loc='upper left',title=r'IPA · 100 $\mu$m · 9 h',fontsize=8.5,title_fontsize=9.5,framealpha=1,fancybox=False,edgecolor='black',handlelength=2.2,handletextpad=.6,labelspacing=.25,borderpad=.35,borderaxespad=.3).get_frame().set_linewidth(.8)
valid=rp[rp.radius_mm.gt(0)];used=valid[valid.used_common];outside=valid[~valid.used_common]
ax=axa;xf=np.array([used.machine_displacement_mm.min(),used.machine_displacement_mm.max()])
ax.plot(xf,rc.K_a*xf+rc.C_a_mm,color='black',ls=(0,(4.5/2.05,3/2.05)),lw=2.05,label='Linear fit')
ax.scatter(used.machine_displacement_mm,used.radius_mm,c='red',s=24,zorder=3)
ax.scatter(outside.machine_displacement_mm,outside.radius_mm,facecolors='white',edgecolors='red',s=24,label='Measured',zorder=3)
ax.set(xlim=(0,10.5),ylim=(-.5,17),xlabel=r'$\delta$ (mm)',ylabel=r'$a$ (mm)',xticks=[0,5,10],yticks=[0,8,16])
ax.legend(loc='upper left',title=r'IPA · 100 $\mu$m · 9 h',fontsize=8.5,title_fontsize=9.5,framealpha=1,fancybox=False,edgecolor='black',handlelength=2.2,handletextpad=.6,labelspacing=.25,borderpad=.35,borderaxespad=.3).get_frame().set_linewidth(.8)
group=force[force.liquid.eq(selection['liquid'])&force.ideal_dry_time_h.eq(selection['conditioning_time_h'])&force.repeat_id.eq(selection['repeat_id'])].sort_values('thickness_um')
assert len(group)==4 and set(group.thickness_um)==set(THICKNESS_COLORS)
inset=axm.inset_axes([.5,.12,.46,.27]);curve_points=[]
for row in group.itertuples():
    d=book.parse(row.test_id);d=d[d['Displacement (mm)'].between(row.fit_curve_region_start_mm,row.fit_curve_region_end_mm)]
    x=d['Displacement (mm)'].to_numpy();y=d['Load (N)'].to_numpy();x0=x.min();y0=row.slope_n_per_mm*x0+row.fit_intercept_n
    dx=x-x0;dy=y-y0;normalized=dy*1000/np.sqrt(row.S_N_per_m)
    color=THICKNESS_COLORS[int(row.thickness_um)]
    axm.plot(dx,normalized,color=color,lw=.75)
    axm.plot([0,dx.max()],[0,row.K_F_N_per_m*dx.max()/np.sqrt(row.S_N_per_m)],color='black',ls=(0,(4.5/2.05,3/2.05)),lw=2.05)
    inset.plot(dx,dy,color=color,lw=.75);inset.plot([0,dx.max()],[0,row.slope_n_per_mm*dx.max()],color='black',ls=(0,(4.5/1.6,3/1.6)),lw=1.6)
    curve_points.extend(dict(test_id=row.test_id,thickness_um=row.thickness_um,delta_mm=a,F_N=b,
        delta_increment_mm=c,force_increment_N=e,normalized_force_milli_sqrt_Nm=f) for a,b,c,e,f in zip(x,y,dx,dy,normalized))
axm.set(xlim=(-.15,10.15),ylim=(-35,60),xlabel=r'$\Delta\delta$ (mm)',ylabel=r'$\Delta F/\sqrt{S}$ ($\sqrt{\mathrm{N}\,\mu\mathrm{m}}$)',xticks=[0,5,10],yticks=[-20,0,20,40,60])
inset.set(xlim=(0,9.8),ylim=(-.025,1.03),xlabel=r'$\Delta\delta$ (mm)',ylabel=r'$\Delta F$ (N)',xticks=[0,4,8],yticks=[0,.5,1]);inset.tick_params(direction='in',top=True,right=True,labelsize=8.5,length=2.5,pad=2)
for spine in inset.spines.values():spine.set_linewidth(.75)
inset.xaxis.label.set_size(9);inset.yaxis.label.set_size(9)
inset.xaxis.labelpad=2;inset.yaxis.labelpad=2
inset.xaxis.label.set_bbox({'facecolor':'white','edgecolor':'none','pad':1.2})
handles=[Line2D([],[],color=THICKNESS_COLORS[int(h)],label=rf'{int(h)} $\mu$m') for h in group.thickness_um]
handles.append(Line2D([],[],color='black',ls=(0,(4.5/2.05,3/2.05)),lw=2.05,label='Linear fits'))
axm.legend(handles=handles,loc='upper left',title='IPA · 9 h',fontsize=8.5,title_fontsize=9.5,framealpha=1,fancybox=False,edgecolor='black',bbox_to_anchor=(.01,.985),handlelength=2.25,labelspacing=.4,handletextpad=.65,borderpad=.35,borderaxespad=.3).get_frame().set_linewidth(.8)
gap=.012;width=(.9-4*gap)/5;height=width*fig.get_figwidth()/fig.get_figheight()
offset=layout['photo_label_offset_pt']
fig.text(.085+offset[0]/(fig.get_figwidth()*72),.058+height+offset[1]/(fig.get_figheight()*72),'(c)',fontsize=12,ha='left',va='baseline')
assert selection['representative_radius_sample']=='ipa_100um_9h','Update the approved photograph selection for this sample.'
images,photos=approved_photos('figure3')
for i,(image,recipe) in enumerate(zip(images,photos)):
    ax=fig.add_axes([.085+i*(width+gap),.058,width,height]);ax.imshow(image,origin='upper',interpolation='none')
    ax.set(xticks=[],yticks=[])
    for spine in ax.spines.values():spine.set_linewidth(.8)
    ax.text(.5,-.065,f"{recipe['time_s']:.2f} s",transform=ax.transAxes,ha='center',va='top',fontsize=9)
book.close();save_publication_figure(fig,OUT/'Fig3_blister',composite=True);plt.close(fig)
pd.DataFrame(curve_points).to_csv(DATA/'fig3_master_curve_points.csv',index=False)
rp.to_csv(DATA/'fig3_representative_radius.csv',index=False)
print('Figure 3 regenerated from raw curves, video and annotations.')
