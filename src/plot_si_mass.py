"""Reproduce SI recorded-mass figures without changing source data.

The portrait layout fills an SI figure page at 0.98 linewidth. Defaults write
PDF/PNG/SVG figures into the delivered generated directory.
All figures use the same mass scale, in grams, and plot mass_g directly.
"""
from pathlib import Path
import argparse
import csv
import json
import hashlib
import math
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from package_paths import ROOT, RAW, CONFIG, PROCESSED, OUTPUT
from figure_settings import SI_COLORS, FONT_FAMILY
LIQUIDS = ("Water", "Ethanol", "IPA", "Glycerol")
COLORS = SI_COLORS
COUNTS = {"Water": (118, 21), "Ethanol": (96, 18),
          "IPA": (79, 17), "Glycerol": (44, 11)}
# Nominal masses for 500 microlitres, consistent with every recorded first reading.
# These values validate the inputs; they never replace a measured mass.
INITIAL_MASS_G = json.loads((CONFIG/"analysis.json").read_text())["mass_initial_g"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_style():
    font_dir = Path("C:/Windows/Fonts")
    for path in font_dir.glob("times*.ttf"):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [FONT_FAMILY, "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "mathtext.fontset": "stix", "axes.linewidth": .65,
        "xtick.direction": "out", "ytick.direction": "out",
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white",
        "path.simplify": False,
    })


def plot_liquid(rows, liquid):
    points = [row for row in rows if row["liquid"] == liquid]
    counts = (len(points), len({r["sequence_id"] for r in points}))
    fig, axes = plt.subplots(4, 2, figsize=(6.8, 7.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.985, bottom=.065, top=.955,
                        hspace=.36, wspace=.17)
    max_time = max(float(r["time_h"]) for r in points)
    xticks = np.arange(0, math.ceil(max_time) + 1, 2)
    # A common scale shows the different initial masses for equal liquid volumes.
    ylim, yticks = (-.02, .75), (0, .2, .4, .6)
    assert all(np.isclose(float(r["mass_g"]), INITIAL_MASS_G[liquid])
               for r in points if int(r["point_order"]) == 1)
    drawn = 0
    for col, workflow in enumerate(("indentation", "peeling")):
        for row_index, thickness in enumerate((25, 50, 100, 150)):
            ax = axes[row_index, col]
            panel_rows = [r for r in points if r["workflow_label"] == workflow
                          and int(r["thickness_um"]) == thickness]
            # Source order also keeps clock-reset segments independent.
            sequence_ids = list(dict.fromkeys(r["sequence_id"] for r in panel_rows))
            for index, sequence_id in enumerate(sequence_ids):
                sequence = [r for r in panel_rows if r["sequence_id"] == sequence_id]
                x = np.array([float(r["time_h"]) for r in sequence])
                y = np.array([float(r["mass_g"]) for r in sequence])
                line, = ax.plot(x, y, "o-", color=COLORS[index % len(COLORS)],
                                lw=.9, ms=2.8, label=f"Sequence {index + 1}")
                assert np.array_equal(line.get_xdata(), x)
                assert np.array_equal(line.get_ydata(), y)
                drawn += len(sequence)
            letter = chr(ord("a") + row_index + 4 * col)
            ax.set_title(f"({letter}) {thickness} $\\mu$m", loc="left", pad=6)
            ax.set_xlim(-max_time * .025, max_time * 1.05)
            ax.set_ylim(*ylim)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.tick_params(length=2.8, width=.6, pad=3)
            ax.grid(axis="y", color="#e5e5e5", lw=.45)
            ax.set_axisbelow(True)
            ax.legend(loc="lower right" if liquid == "Glycerol" else "upper right",
                      frameon=False, ncol=2 if len(sequence_ids) > 4 else 1,
                      handlelength=1.5, handletextpad=.5, columnspacing=.8)
            if col == 0:
                ax.set_ylabel(r"$m$ (g)")
            if row_index == 3:
                ax.set_xlabel("Recorded time (h)", labelpad=5)
        axes[0, col].text(.5, 1.24,
            "Blister-test preparation" if col == 0 else "Peeling preparation",
            transform=axes[0, col].transAxes, ha="center", va="bottom")
    assert drawn == len(points)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = PROCESSED/"mass_points.csv"
    protected = [source, PROCESSED/"mass_sequences.csv",
                 RAW/"mass/mass_indentation.xlsx",
                 RAW/"mass/mass_peeling.xlsx"]
    before = {p: digest(p) for p in protected}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and len({r["sequence_id"] for r in rows}) > 0
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for liquid in LIQUIDS:
        fig = plot_liquid(rows, liquid)
        stem = args.output_dir / f"weight_{liquid.lower()}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.04,
                    metadata={"Title": f"Recorded mass during {liquid.lower()} conditioning", "CreationDate": None, "ModDate": None})
        plt.close(fig)
        print(f"{liquid}: {len([r for r in rows if r["liquid"] == liquid])} points, {len({r["sequence_id"] for r in rows if r["liquid"] == liquid})} sequences")
    assert before == {p: digest(p) for p in protected}, "Source data changed"


if __name__ == "__main__":
    main()
