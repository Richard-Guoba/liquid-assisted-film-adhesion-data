"""Replay manual annotations against original videos; never redraw markers."""
from pathlib import Path
import json,re,math
import numpy as np
import pandas as pd
import cv2
import radius_process
from radius_prepare import normalize_selection_ranges
from calculate import linear,save

def last_token(value):
    if pd.isna(value):return None
    return re.split(r'[\s,;，；/]+',str(value).strip())[-1]

def align_points(sample,selected,metadata,raw,video):
    data=raw[['Time (s)','Displacement (mm)']].apply(pd.to_numeric,errors='coerce').dropna()
    times=data['Time (s)'].to_numpy();displacement=data['Displacement (mm)'].to_numpy()
    vtoken=last_token(sample['原视频时间']);ftoken=last_token(sample['力名义时间(大约)'])
    if str(sample.get('新对齐','')).lower()=='begin':
        keep=displacement>=float(metadata.fit_curve_region_start_mm)
        offset=float(times[np.flatnonzero(keep)[0]])
        source='video zero aligned to configured force-fit start'
    elif vtoken is not None and ftoken is not None:
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);last=(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1)/fps;cap.release()
        vtime=last if vtoken.lower()=='end' else float(vtoken)
        if ftoken.lower()=='end':ftime=float(times[-1])
        else:
            target=float(ftoken)*float(metadata.velocity_mm_min)/60
            table=data.groupby('Displacement (mm)',as_index=False)['Time (s)'].mean().sort_values('Displacement (mm)')
            ftime=float(np.interp(target,table['Displacement (mm)'],table['Time (s)']))
        offset=ftime-vtime;source='last recorded video/nominal-force event; actual machine-time interpolation'
    else:
        if selected.empty:raise ValueError('no manually annotated frame selection supplied')
        result=selected.copy()
        result['machine_time_s']=np.interp(result.displacement_mm,displacement,times)
        result['alignment_source']='archived manual frame/displacement pairing; independent event alignment unavailable'
        result['video_to_machine_offset_s']=np.nan
        return result
    result=selected.copy()
    result['machine_time_s']=result.time_s+offset
    result['displacement_mm']=np.interp(result.machine_time_s,times,displacement)
    result['alignment_source']=source
    result['video_to_machine_offset_s']=offset
    return result

def calculate_radius(root,out,settings,force):
    selection=pd.read_csv(root/'config/radius_selected_frames.csv')
    config=normalize_selection_ranges(pd.read_excel(root/'config/radius_config.xlsx',sheet_name='selection_ranges'))
    fc=pd.read_excel(root/'config/force_config.xlsx').set_index('test_id')
    forcebook=pd.ExcelFile(root/'data/force/force_data.xlsx')
    all_points=[];all_results=[];common=[]
    for index,row in config.iterrows():
        sample=str(row['sample']);ident=str(row.force_sheet);metadata=fc.loc[ident]
        video=root/'data/radius'/str(row.video_file)
        local=selection[selection['sample'].eq(sample)].copy()
        try:
            if not video.is_file():raise FileNotFoundError(str(video))
            aligned=align_points(row,local,metadata,forcebook.parse(ident),video)
            result,points=radius_process.process_sample(row,aligned,metadata,root/'data/radius',root/'config/radius_annotations',settings['film_diameter_mm'])
            if len(points):
                points=points.rename(columns={'displacement_mm':'machine_displacement_mm'})
                points['force_test_id']=ident
                points['alignment_source']=aligned.alignment_source.iloc[0]
                points['used_common']=points.radius_mm.gt(0)&points.machine_displacement_mm.between(metadata.fit_curve_region_start_mm,metadata.fit_curve_region_end_mm)
                all_points.append(points)
                used=points[points.used_common].dropna(subset=['radius_mm','machine_displacement_mm'])
                if len(used)>=3:
                    ka,ca,r2=linear(used.machine_displacement_mm.to_numpy(),used.radius_mm.to_numpy())
                    f=force[force.test_id.eq(ident)].iloc[0]
                    raw=forcebook.parse(ident)
                    lo,hi=used.machine_displacement_mm.min(),used.machine_displacement_mm.max()
                    cut=raw[raw['Displacement (mm)'].between(lo,hi)]
                    kf,cf,_=linear(cut['Displacement (mm)'].to_numpy(),cut['Load (N)'].to_numpy())
                    A0=1+(1-3*settings['poisson_ratio'])/(3*(5-3*settings['poisson_ratio']))
                    gf=3*A0**3/(4*math.pi**2)*(1000*kf)**2/f.S_N_per_m
                    ga=f.S_N_per_m/(12*A0**3*ka**4) if ka>0 else None
                    common.append(dict(specimen_id=ident,radius_sample=sample,liquid=f.liquid,thickness_um=f.thickness_um,
                        conditioning_time_h=f.ideal_dry_time_h,n_common_radii=len(used),window_start_mm=lo,window_end_mm=hi,
                        K_a=ka,C_a_mm=ca,radius_R2=r2,K_F_common_N_per_m=kf*1000,C_F_common_N=cf,
                        K_F_reported_N_per_m=f.K_F_N_per_m,C_F_reported_N=f.fit_intercept_n,
                        force_reported_start_mm=f.fit_curve_region_start_mm,force_reported_end_mm=f.fit_curve_region_end_mm,
                        Gamma_F_common=gf,Gamma_B_reported=f.Gamma_B_J_m2,Gamma_a_slope=ga,
                        Gamma_a_over_Gamma_F_common=ga/gf if ga is not None and gf>0 else None,
                        alignment_independently_verified=False))
            result['force_test_id']=ident;all_results.append(result)
        except Exception as error:
            all_results.append(dict(sample=sample,force_test_id=ident,status='unavailable',failure_reason=str(error)))
        print(f'Radius {index+1}/{len(config)}: {sample} {all_results[-1]["status"]}',flush=True)
    forcebook.close()
    known=set(config.force_sheet)
    for row in force[~force.test_id.isin(known)].itertuples():
        all_results.append(dict(sample=None,force_test_id=row.test_id,status='no_video_or_annotations_supplied',failure_reason='force-based adhesion is available; radius measurement is unavailable'))
    points=pd.concat(all_points,ignore_index=True) if all_points else pd.DataFrame()
    results=pd.DataFrame(all_results);common=pd.DataFrame(common)
    save(points,out/'common_points.csv');save(results,out/'radius_results.csv');save(common,out/'radius_common.csv')
    return results,common
