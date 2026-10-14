"""Download the original videos from GitHub Release and verify every SHA256."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def fetch(record):
    target=ROOT/'data/radius'/record['name']
    if target.exists() and target.stat().st_size==record['size_bytes'] and digest(target)==record['sha256']:return record['name']+' verified'
    temporary=target.with_suffix('.mp4.download')
    req=urllib.request.Request(record['download_url'],headers={'User-Agent':'liquid-assisted-film-adhesion-data'})
    with urllib.request.urlopen(req,timeout=120) as response,temporary.open('wb') as f:
        for block in iter(lambda:response.read(1024*1024),b''):f.write(block)
    if temporary.stat().st_size!=record['size_bytes'] or digest(temporary)!=record['sha256']:
        temporary.unlink();raise ValueError('Video checksum mismatch: '+record['name'])
    temporary.replace(target);return record['name']+' downloaded and verified'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=int,default=3);args=p.parse_args()
    records=json.loads((ROOT/'data/radius/video_manifest.json').read_text(encoding='utf-8'))['videos']
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for status in pool.map(fetch,records):print(status,flush=True)
if __name__=='__main__':main()
