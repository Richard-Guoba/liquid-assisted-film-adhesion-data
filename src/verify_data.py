"""Check the quantities in the actual Figure 4 PDF against all computed trials."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from pypdf import PdfReader

def verify(root,out):
    data=out/'figures/data';master=pd.read_csv(out/'processed/specimen_master.csv')
    points=pd.read_csv(data/'Fig4_plot_points.csv');means=pd.read_csv(data/'Fig4_bar_statistics.csv')
    source=master.set_index('specimen_id');plotted=points.set_index('sample_id')
    assert set(source.index)==set(plotted.index) and plotted.index.is_unique
    assert np.allclose(source.Gamma_current,plotted.loc[source.index].Gamma_J_m2,rtol=1e-12,atol=1e-12)
    for row in means.itertuples():
        for method,field,nfield in [('Blister','mean_Gamma_B_J_m2','number_of_blister_records'),('Peeling','mean_Gamma_P_J_m2','number_of_peeling_records')]:
            z=points[points.liquid.eq(row.liquid)&points.method.eq(method)&points.display_nominal_group_h.eq(row.nominal_time_h)]
            assert len(z)==getattr(row,nfield) and np.isclose(z.Gamma_J_m2.mean(),getattr(row,field),rtol=1e-12)
    page=PdfReader(out/'figures/Fig4_all_adhesion.pdf').pages[0];marks=[];bars=[];state={'clip':None,'color':None,'path':[]};legend=[]
    def operand(op,a,cm,tm):
        if op==b're':state['clip']=[float(v) for v in a]
        if op==b'rg':state['color']=[float(v) for v in a]
        if op==b'g':state['color']=[float(a[0])]*3
        if op==b'Do' and str(a[0]).startswith('/M'):marks.append({'resource':str(a[0]),'x':cm[4],'y':cm[5],'clip':state['clip']})
        if op==b'm':state['path']=[]
        if op in [b'm',b'l']:
            x,y=map(float,a);state['path'].append((cm[0]*x+cm[2]*y+cm[4],cm[1]*x+cm[3]*y+cm[5]))
        if op in [b'B',b'f',b'f*',b'B*',b'b',b'b*']:
            color,path,clip=state['color'],state['path'],state['clip']
            if color and len(path)==4 and max(color)-min(color)<1e-10 and .2<color[0]<.96 and clip:
                xs,ys=zip(*path)
                if abs(min(ys)-clip[1])<1e-5 and clip[0]-.001<=min(xs)<max(xs)<=clip[0]+clip[2]+.001 and max(ys)<=clip[1]+clip[3]+.001:
                    bars.append({'center':(min(xs)+max(xs))/2,'y':max(ys),'clip':clip})
            state['path']=[]
    def text(t,cm,tm,font,size):
        if 'Peeling mean' in t:legend.append(cm[1]*tm[4]+cm[3]*tm[5]+cm[5])
    page.extract_text(visitor_operand_before=operand,visitor_text=text)
    assert legend and all(0<=y<=float(page.mediabox.height) for y in legend)
    settings=json.loads((root/'config/figure_selection.json').read_text(encoding='utf-8'))
    xmin,xmax=-.48,2.48;ymin,ymax=0,max(settings['figure4']['minimum_y_max'],float(points.Gamma_J_m2.max()*1.05))
    assert len(marks)==len(master)
    for group in range(32):
        liquid=['Ethanol','IPA','Water','Glycerol'][group//8];method=['Blister','Peeling'][(group%8)//4];h=[25,50,100,150][group%4]
        actual=sorted(((m['x']-m['clip'][0])/m['clip'][2]*(xmax-xmin)+xmin,(m['y']-m['clip'][1])/m['clip'][3]*(ymax-ymin)+ymin) for m in marks if m['resource']==f'/M{group}')
        z=points[points.liquid.eq(liquid)&points.method.eq(method)&points.thickness_um.eq(h)]
        wanted=sorted(zip((z.display_nominal_group_h-3)/3+z.display_offset,z.Gamma_J_m2))
        assert len(actual)==len(wanted) and np.allclose(actual,wanted,rtol=0,atol=1e-7)
    assert len(bars)==2*len(means)
    panels=sorted(set(b['clip'][0] for b in bars))
    for b in bars:
        x=(b['center']-b['clip'][0])/b['clip'][2]*(xmax-xmin)+xmin;gamma=(b['y']-b['clip'][1])/b['clip'][3]*(ymax-ymin)+ymin
        liquid=['Ethanol','IPA','Water','Glycerol'][panels.index(b['clip'][0])];time=round(x)*3+3
        row=means[means.liquid.eq(liquid)&means.nominal_time_h.eq(time)].iloc[0]
        assert abs(gamma-row['mean_Gamma_B_J_m2' if x-round(x)<0 else 'mean_Gamma_P_J_m2'])<1e-7
    return {'adhesion_trials':len(master),'actual_pdf_points':len(marks),'actual_pdf_mean_bars':len(bars),'legend_inside_page':True}
