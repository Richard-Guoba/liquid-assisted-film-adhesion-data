"""Recalculate calibration, adhesion and weighing tables from raw records."""
from pathlib import Path
from dataclasses import asdict
import json, math, re
import numpy as np
import pandas as pd
import force_process, peeling_process, modulus_process

def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g')

def linear(x,y):
    k,b=np.polyfit(x,y,1);r=y-(k*x+b);ss=np.sum((y-y.mean())**2)
    return float(k),float(b),float(1-np.sum(r*r)/ss) if ss else 1.0

def calibration(root, out, settings):
    configs=modulus_process.load_modulus_config(root/'config/modulus_config.xlsx')
    book=pd.ExcelFile(root/'data/modulus/modulus_data.xlsx')
    records=[];a=settings['calibration_true_strain_start'];b=settings['calibration_true_strain_end']
    for c in configs:
        curve=modulus_process.convert_curve(c,book.parse(c.test_id))
        x,y=curve.true_strain,curve.true_stress_mpa
        complete=float(x.min())<=a and float(x.max())>=b
        mask=(x>=a)&(x<=b)
        included=bool(complete and mask.sum()>=2 and np.ptp(x[mask])>0)
        row=dict(record_id=c.test_id,specimen_type=c.specimen_type,nominal_thickness_um=c.thickness_um,
            gauge_length_mm=c.gauge_length_mm,width_mm=c.width_mm,displacement_zero_mm=curve.displacement_zero_mm,
            raw_true_strain_min=float(x.min()),raw_true_strain_max=float(x.max()),included=included,
            fit_point_count=int(mask.sum()),inclusion_reason='complete_true_strain_interval' if included else 'incomplete_true_strain_interval')
        if included:
            E,intercept,r2=linear(x[mask],y[mask]);row.update(E_fit_MPa=E,S_fit_N_per_m=E*c.thickness_um,
                fit_intercept_MPa=intercept,fit_R2=r2,actual_fit_min=float(x[mask].min()),actual_fit_max=float(x[mask].max()))
        records.append(row)
    book.close();records=pd.DataFrame(records);kept=records[records.included]
    summary=kept.groupby('nominal_thickness_um').agg(n=('record_id','size'),E_mean_MPa=('E_fit_MPa','mean'),
        E_SD_MPa=('E_fit_MPa','std'),S_N_per_m=('S_fit_N_per_m','mean'),S_SD_N_per_m=('S_fit_N_per_m','std')).reset_index().rename(columns={'nominal_thickness_um':'thickness_um'})
    assert set(summary.thickness_um)=={25,50,100,150}
    save(records,out/'modulus_records.csv');save(summary,out/'modulus_summary.csv')
    return summary

def adhesion(root,out,summary,settings):
    force=force_process.results_frame(force_process.compute_force_results(root/'data/force/force_data.xlsx',root/'config/force_config.xlsx'))
    peel=peeling_process.process_workbook(root/'data/peeling/peeling_data.xlsx',root/'config/peeling_config.xlsx')
    force['liquid']=force.liquid.str.lower().replace({'oil':'glycerol'})
    peel['liquid']=peel.liquid.str.lower().replace({'oil':'glycerol'})
    force=force.merge(summary[['thickness_um','S_N_per_m','E_mean_MPa']],on='thickness_um',validate='many_to_one')
    nu=float(settings['poisson_ratio']);A0=1+(1-3*nu)/(3*(5-3*nu));coeff=3*A0**3/(4*math.pi**2)
    force['K_F_N_per_m']=force.slope_n_per_mm*1000
    force['Gamma_B_J_m2']=coeff*force.K_F_N_per_m**2/force.S_N_per_m
    force['quality']=np.where((force.r_squared>=settings['qc']['force_min_r2'])&(force.slope_n_per_mm>0),'linear_fit','review_required')
    rows=[]
    for r in force.itertuples():
        rows.append(dict(specimen_id=r.test_id,method='uplift',liquid=r.liquid,thickness_um=r.thickness_um,
            conditioning_time_h=r.ideal_dry_time_h,repeat_id=r.repeat_id,S_N_per_m=r.S_N_per_m,E_MPa=r.E_mean_MPa,
            K_F_N_per_m=r.K_F_N_per_m,C_F_N=r.fit_intercept_n,fit_R2=r.r_squared,
            interval_start_mm=r.fit_curve_region_start_mm,interval_end_mm=r.fit_curve_region_end_mm,
            Gamma_current=r.Gamma_B_J_m2,Gamma_primary=r.Gamma_B_J_m2,primary_eligible=True,
            quality=r.quality,raw_sheet=r.test_id,raw_source='data/force/force_data.xlsx'))
    quality=[];book=pd.ExcelFile(root/'data/peeling/peeling_data.xlsx')
    for r in peel.itertuples():
        d=peeling_process.prepare_curve(book.parse(r.sheet_name));z=d[d['Displacement zeroed (mm)'].between(r.plateau_start_mm_zeroed,r.plateau_end_mm_zeroed)]
        x=z['Displacement zeroed (mm)'].to_numpy();y=z['Load (N)'].to_numpy();k,b,_=linear(x,y)
        cv=float(np.std(y,ddof=1)/abs(y.mean()));drift=float(k*(x.max()-x.min())/abs(y.mean()))
        mad=np.median(np.abs(y-np.median(y)));spikes=float(np.mean(np.abs(y-np.median(y))>6*1.4826*mad)) if mad else 0
        quality.append(dict(sheet_name=r.sheet_name,plateau_cv=cv,relative_linear_drift=drift,
            plateau_spike_fraction_6MAD=spikes))
        flag='review_required' if cv>settings['qc']['peeling_max_cv'] or abs(drift)>settings['qc']['peeling_max_relative_drift'] else 'configured_mean'
        rows.append(dict(specimen_id=r.sheet_name,method='peeling',liquid=r.liquid,thickness_um=r.thickness_um,
            conditioning_time_h=r.drying_time_h,repeat_id=None,S_N_per_m=None,E_MPa=None,
            K_F_N_per_m=None,C_F_N=None,fit_R2=None,interval_start_mm=r.plateau_start_mm_zeroed,
            interval_end_mm=r.plateau_end_mm_zeroed,Gamma_current=r.gamma_n_per_m,Gamma_primary=r.gamma_n_per_m,
            primary_eligible=True,quality=flag,raw_sheet=r.sheet_name,raw_source='data/peeling/peeling_data.xlsx'))
    book.close();master=pd.DataFrame(rows)
    peel=peel.merge(pd.DataFrame(quality),on='sheet_name',validate='one_to_one')
    save(force,out/'force_fit_results.csv');save(peel,out/'peeling_results.csv');save(master,out/'specimen_master.csv')
    cells=master.groupby(['method','liquid','thickness_um','conditioning_time_h']).agg(mean_Gamma=('Gamma_current','mean'),n=('specimen_id','size')).reset_index()
    stats=master.groupby(['method','liquid']).agg(n=('specimen_id','size'),median_Gamma=('Gamma_current','median'),mean_Gamma=('Gamma_current','mean')).reset_index()
    cv=[]
    for liquid in ['ethanol','ipa']:
        for time in [3,6,9]:
            records=force[force.liquid.eq(liquid)&force.ideal_dry_time_h.eq(time)]
            groups=records.groupby('thickness_um').agg(K=('K_F_N_per_m','mean'),S=('S_N_per_m','first'))
            assert len(groups)==4
            values=groups.K.to_numpy();normalized=values/np.sqrt(groups.S.to_numpy())
            cv.append(dict(scope='all_measured_force_curves',liquid=liquid,time_h=time,n_records=len(records),n_thickness_cells=4,
                raw_CV=np.std(values)/np.mean(values),normalized_CV=np.std(normalized)/np.mean(normalized)))
    save(cells,out/'adhesion_cells.csv');save(stats,out/'adhesion_summary.csv');save(pd.DataFrame(cv),out/'normalization_CV.csv')
    return force,peel,master

def mass(root,out):
    points=[];sequences=[];order=0
    for workflow,name in [('indentation','mass_indentation.xlsx'),('peeling','mass_peeling.xlsx')]:
        source=pd.read_excel(root/'data/mass'/name,sheet_name='data')
        source['source_excel_row']=np.arange(len(source))+2
        for (liquid,h,rep),group in source.groupby(['liquid','thickness_um','replicate'],sort=False):
            segment=1;current=[]
            def flush(current,segment):
                nonlocal order
                order+=1;sid=f'{workflow}:{str(liquid).lower()}:{int(h)}um:rep{int(rep)}:segment{segment}'
                for i,row in enumerate(current,1):
                    points.append(dict(sequence_id=sid,sequence_order=order,source_file=f'data/mass/{name}',source_sheet='data',
                        source_excel_row=int(row.source_excel_row),workflow_label=workflow,liquid=liquid,thickness_um=h,
                        recorded_replicate=rep,point_order=i,time_h=float(row.time_h),mass_g=float(row.mass_g)))
                delta=float(current[-1].mass_g-current[-2].mass_g) if len(current)>1 else None
                sequences.append(dict(sequence_id=sid,workflow_label=workflow,liquid=liquid,thickness_um=h,n_readings=len(current),
                    first_time_h=float(current[0].time_h),last_time_h=float(current[-1].time_h),last_recorded_mass_g=float(current[-1].mass_g),
                    terminal_pair_delta_mass_g=delta,terminal_pair_recorded_change='decreased' if delta is not None and delta<0 else 'increased' if delta is not None and delta>0 else 'unchanged_recorded_value',
                    independent_specimen_id_known=False,dry_mass_reference_known=False))
            for row in group.itertuples():
                if current and row.time_h<current[-1].time_h:
                    flush(current,segment);current=[];segment+=1
                current.append(row)
            if current:flush(current,segment)
    points=pd.DataFrame(points);sequences=pd.DataFrame(sequences)
    save(points,out/'mass_points.csv');save(sequences,out/'mass_sequences.csv')
    assert len(points)==sum(len(pd.read_excel(root/'data/mass'/name)) for name in ['mass_indentation.xlsx','mass_peeling.xlsx'])
    return points,sequences

def calculate_all(root,out):
    settings=json.loads((root/'config/analysis.json').read_text());out.mkdir(parents=True,exist_ok=True)
    summary=calibration(root,out,settings)
    force,peel,master=adhesion(root,out,summary,settings)
    points,sequences=mass(root,out)
    print(f'Calculated {len(force)} blister, {len(peel)} peeling, {len(points)} mass readings, {len(sequences)} weighing sequences.',flush=True)
    return settings,summary,force,peel,master
