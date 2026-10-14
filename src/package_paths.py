"""Portable input/output locations. No cached scientific results are inputs."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data'
CONFIG = ROOT / 'config'
ASSETS = RAW / 'photographs'
OUTPUT_ROOT = Path(os.environ.get('SUBMISSION_OUTPUT_ROOT', str(ROOT/'results'))).resolve()
PROCESSED = OUTPUT_ROOT / 'processed'
OUTPUT = OUTPUT_ROOT / '.work' / 'figures'
os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT_ROOT/'.work/matplotlib'))
