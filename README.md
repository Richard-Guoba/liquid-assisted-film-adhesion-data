# Liquid-conditioned film adhesion

Experimental data and analysis code for liquid-conditioned PDMS films on glass.

| Directory | Contents |
| --- | --- |
| `data/` | Raw workbooks, original videos and figure photographs |
| `config/` | Specimen conditions, fitting windows, radius annotations, selected frames and analysis/figure settings |
| `src/` | Analysis and plotting code |
| `results/processed/` | Stiffness, force fits, adhesion, radii and mass results, including fit quality and inclusion flags |
| `results/figures/` | Figures 2-4 and four SI mass figures |
| `results/figures/data/` | Values and curves shown in the figures |

The workbooks contain 69 blister trials, 55 peeling trials, 53 tensile records and 337 mass readings. Use Python 3.12:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -r requirements.txt
python fetch_videos.py
python run_analysis.py
```

On Linux/macOS, activate with `source .venv/bin/activate`. The 53 videos are stored in [release data-v1.0.0](https://github.com/Richard-Guoba/liquid-assisted-film-adhesion-data/releases/tag/data-v1.0.0); `fetch_videos.py` downloads them and verifies their hashes. It can be skipped when the videos are already present.

The workflow reads raw data and configuration, calculates results, then plots them. It checks all 124 values and 24 mean bars in the Figure 4 PDF and leaves the inputs unchanged. TeX Live with `latex`, `dvipng` and `newtx` reproduces the figure typography; Matplotlib text is used when TeX is unavailable.

To save a separate run or check existing outputs:

```bash
python run_analysis.py --output-dir results_rerun
python run_analysis.py --stage verify
```

Time, force and displacement are in s, N and mm; configuration speeds are in mm/min. Blister fits use recorded machine displacement and a free intercept. Peeling windows use displacement relative to the first valid reading, and adhesion is mean plateau force divided by strip width in metres.

Stiffness uses the 29 of 53 tensile records spanning the full 10%-20% true-strain interval. `E` is in MPa, `S=Eh` in N/m and `Gamma_current` in J/m². In result tables, `uplift` means blister; video filenames using `oil` refer to glycerol. Conditions are defined by the configuration workbooks, and groups use nominal conditioning times.

Radius configuration uses English field names. `selection_signature.txt` binds each annotation set to its selected frames. Keep it with the annotations. Sixteen blister trials have no videos; missing or geometrically invalid radii remain unavailable. Figure photographs retain the author's markings, while radii are calculated from separate annotations.

The membrane model uses crosshead displacement as a height proxy and stiffness measured before liquid conditioning. All configured adhesion trials are included, with fit and plateau-quality flags retained. The mass records have no dry-mass reference or specimen-level link to the adhesion trials.
