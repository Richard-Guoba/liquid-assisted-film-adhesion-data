"""Original TeX typography when available; portable Matplotlib otherwise."""
import os, shutil, subprocess
from figure_settings import FONT_FAMILY
def available_tex():
    if not all(shutil.which(name) for name in ('latex','dvipng','kpsewhich')):
        return False
    for name in ('newtxtext.sty','newtxmath.sty'):
        result=subprocess.run(['kpsewhich',name],capture_output=True,text=True)
        if result.returncode or not result.stdout.strip(): return False
    return True
mode=os.environ.get('FIGURE_TEXT_ENGINE','auto')
USE_TEX=mode=='tex' or (mode=='auto' and available_tex())
if mode=='tex' and not available_tex():
    raise RuntimeError('TeX mode needs latex, dvipng, newtxtext and newtxmath. Set FIGURE_TEXT_ENGINE=matplotlib to use the fallback.')
TEXT_RC={'text.usetex':USE_TEX,'font.family':'serif' if USE_TEX else FONT_FAMILY,
         'text.latex.preamble':r'\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}'}
def formatted(value):
    if USE_TEX: return value
    value=value.replace(r'\(','$').replace(r'\)','$')
    if value.startswith(r'\textbf{') and value.endswith('}'):
        value=value[len(r'\textbf{'):-1]
    return value
