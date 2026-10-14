"""Reproduce experimental calculations and data figures independently of a manuscript."""
from pathlib import Path
import argparse,hashlib,json,os,runpy,shutil,sys,tempfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def input_hashes():
    files=[p for name in ['data','config','src'] for p in sorted((ROOT/name).rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT/'run_analysis.py',ROOT/'requirements.txt']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','plots','verify'],default='all')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results')
    args=p.parse_args();out=args.output_dir.resolve()
    assert out!=ROOT and not any(out.is_relative_to(ROOT/x) for x in ['data','config','src','assets'])
    out.mkdir(parents=True,exist_ok=True)
    if args.stage=='verify':
        from verify_data import verify
        print(json.dumps(verify(ROOT,out),ensure_ascii=False));return
    manifest=json.loads((ROOT/'data/radius/video_manifest.json').read_text(encoding='utf-8'))
    if args.stage=='all':
        missing=[x['name'] for x in manifest['videos'] if not (ROOT/'data/radius'/x['name']).is_file()]
        if missing:raise RuntimeError('Download the raw videos first: python fetch_videos.py')
    before=input_hashes();work=out/'.work';(work/'tmp').mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(work/'tmp')
    try:
        if args.stage=='all':
            from calculate import calculate_all
            from calculate_radius import calculate_radius
            settings,summary,force,peel,master=calculate_all(ROOT,out/'processed')
            calculate_radius(ROOT,out/'processed',settings,force)
        os.environ['SUBMISSION_OUTPUT_ROOT']=str(out);os.environ['MPLCONFIGDIR']=str(work/'matplotlib');os.environ['PYTHONDONTWRITEBYTECODE']='1'
        figures=out/'figures';figures.mkdir(exist_ok=True)
        previous=sys.argv[:]
        try:
            for name in ['plot_fig2.py','plot_fig3.py','plot_fig4.py','plot_si_mass.py']:
                print('Generating',name,flush=True);sys.argv=[name,'--output-dir',str(figures)]
                runpy.run_path(str(ROOT/'src'/name),run_name='__main__')
        finally:sys.argv=previous
        from verify_data import verify
        print(json.dumps(verify(ROOT,out),ensure_ascii=False),flush=True)
        assert before==input_hashes(),'An analysis input changed during the run'
    finally:
        tempfile.tempdir=None
        assert work.resolve().is_relative_to(out);shutil.rmtree(work)
    print('Experimental data reproduced:',out,flush=True)

if __name__=='__main__':main()
