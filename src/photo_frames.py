from pathlib import Path
import json,hashlib
import cv2,numpy as np,pandas as pd
from PIL import Image
from package_paths import ROOT,CONFIG,PROCESSED
from radius_prepare import normalize_selection_ranges

def approved_photos(figure):
    """Load the author's chosen photograph pixels, separately from radius measurements."""
    layout=json.loads((CONFIG/'figure_layout.json').read_text(encoding='utf-8'))
    images=[];recipes=[]
    for record in layout['photographs'][figure]:
        path=ROOT/record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],record['file']
        images.append(np.asarray(Image.open(path).convert('RGB')))
        recipes.append(dict(record,source='author-specified '+figure+'.pdf photograph'))
    return images,recipes

def photo_frame(sample,frame):
    configs=normalize_selection_ranges(pd.read_excel(CONFIG/'radius_config.xlsx',sheet_name='selection_ranges'))
    row=configs[configs['sample'].eq(sample)].iloc[0]
    path=ROOT/'data/radius'/str(row.video_file)
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,image=cap.read();cap.release()
    if not ok:raise ValueError(f'Cannot decode {path.name}, frame {frame}')
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results=pd.read_csv(PROCESSED/'radius_results.csv');cal=results[results['sample'].eq(sample)].iloc[0]
    if pd.notna(cal.get('film_radius_px')):
        cx,cy,r=float(cal.calibration_center_x_px),float(cal.calibration_center_y_px),float(cal.film_radius_px)*1.08
    else:cx,cy,r=image.shape[1]/2,image.shape[0]/2,min(image.shape[:2])/2
    left=max(0,int(cx-r));right=min(image.shape[1],int(cx+r))
    top=max(0,int(cy-r));bottom=min(image.shape[0],int(cy+r))
    return image[top:bottom,left:right],{'source':f'data/radius/{row.video_file}','frame':int(frame),
        'time_s':float(frame/fps),'crop_pixels':[left,top,right,bottom]}

def figure4_photos():
    configs=normalize_selection_ranges(pd.read_excel(CONFIG/'radius_config.xlsx',sheet_name='selection_ranges'))
    fc=pd.read_excel(CONFIG/'force_config.xlsx').set_index('test_id')
    candidates=[r for r in configs.itertuples() if str(fc.loc[r.force_sheet,'liquid']).lower() in ['glycerol','oil']
        and float(fc.loc[r.force_sheet,'thickness_um'])==150 and float(fc.loc[r.force_sheet,'ideal_dry_time_h'])==9]
    row=candidates[0];cap=cv2.VideoCapture(str(ROOT/'data/radius'/str(row.video_file)));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    images=[];recipes=[]
    for time in [10,15]:
        image,recipe=photo_frame(row.sample,round(time*fps))
        images.append(image);recipes.append(recipe)
    return images,recipes
