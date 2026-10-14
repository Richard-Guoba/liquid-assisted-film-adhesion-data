"""All current computed adhesion estimates; no copied result table is read."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from package_paths import CONFIG,PROCESSED,OUTPUT
from paper_style import setup_publication_style,save_publication_figure,adhesion_figure,ADHESION_METHOD_OFFSETS
from text_style import TEXT_RC
from photo_frames import approved_photos

parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=OUTPUT);args=parser.parse_args()
OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True);DATA=OUT/'data';DATA.mkdir(exist_ok=True)
master=pd.read_csv(PROCESSED/'specimen_master.csv');points=[];means=[]
names={'ethanol':'Ethanol','ipa':'IPA','water':'Water','glycerol':'Glycerol'}
offsets={25:-.09,50:-.03,100:.03,150:.09}
for liquid in names:
    local=master[master.liquid.eq(liquid)]
    for row in local.itertuples():
        method='Blister' if row.method=='uplift' else 'Peeling'
        peers=local[local.method.eq(row.method)&local.thickness_um.eq(row.thickness_um)&local.conditioning_time_h.eq(row.conditioning_time_h)]
        ids=list(peers.specimen_id);extra=np.linspace(-.008,.008,len(ids))[ids.index(row.specimen_id)] if len(ids)>1 else 0
        dx=ADHESION_METHOD_OFFSETS[method]+offsets[int(row.thickness_um)]+extra
        points.append(dict(liquid=names[liquid],sample_id=row.specimen_id,method=method,thickness_um=row.thickness_um,
            exact_time_h=row.conditioning_time_h,display_nominal_group_h=row.conditioning_time_h,display_offset=dx,
            Gamma_J_m2=row.Gamma_current,protocol_differs=False,quality=row.quality))
    for time in [3,6,9]:
        b=local[local.method.eq('uplift')&local.conditioning_time_h.eq(time)]
        p=local[local.method.eq('peeling')&local.conditioning_time_h.eq(time)]
        means.append(dict(liquid=names[liquid],nominal_time_h=time,mean_Gamma_B_J_m2=b.Gamma_current.mean(),
            mean_Gamma_P_J_m2=p.Gamma_current.mean(),number_of_blister_records=len(b),number_of_peeling_records=len(p)))
points=pd.DataFrame(points);means=pd.DataFrame(means)
assert len(points)==len(master)
photos,recipes=approved_photos('figure4')
setup_publication_style()
ymax=max(json.loads((CONFIG/'figure_selection.json').read_text(encoding='utf-8'))['figure4']['minimum_y_max'],float(points.Gamma_J_m2.max()*1.05))
with plt.rc_context(TEXT_RC):
    fig=adhesion_figure(points,means,ylim=(0,ymax),insets=[np.flipud(x) for x in photos])
    save_publication_figure(fig,OUT/'Fig4_all_adhesion');plt.close(fig)
points.to_csv(DATA/'Fig4_plot_points.csv',index=False);means.to_csv(DATA/'Fig4_bar_statistics.csv',index=False)
print(f'Figure 4 regenerated: {len(points)} computed estimates.')
