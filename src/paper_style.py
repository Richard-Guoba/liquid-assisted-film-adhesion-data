"""Shared display-only journal style; no data transformations or fitting."""
from pathlib import Path
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.lines import Line2D
from matplotlib.collections import Collection
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from PIL import Image

from figure_settings import FONT_FAMILY, THICKNESS_COLORS as EDIT_COLORS, MEAN_COLORS
from text_style import formatted
from package_paths import CONFIG

INK = '#000000'
HOUSE_FONT = FONT_FAMILY
HOUSE_AXIS_LABEL = 10.5
HOUSE_TICK_LABEL = 9.5
HOUSE_LEGEND = 8.5
HOUSE_CONDITION = 9.5
HOUSE_PANEL = 12
HOUSE_FRAME = .9
HOUSE_LEGEND_FRAME = .8
LIQUID_COLORS = {'Ethanol': '#A65F38', 'IPA': '#355F7C',
                 'Water': '#528C86', 'Glycerol': '#865B83'}
THICKNESS_COLORS = dict(zip((25, 50, 100, 150),
                            ('#355F7C', '#A65F38', '#528C86', '#865B83')))
ADHESION_THICKNESS_COLORS = EDIT_COLORS
THICKNESS_MARKERS = dict(zip((25, 50, 100, 150), ('o', 's', '^', 'D')))
SERIES_STYLES = ('-', (0, (5, 2)), (0, (2, 1.5)), (0, (5, 1.5, 1, 1.5)))
THICKNESS_LINES = dict(zip((25, 50, 100, 150), SERIES_STYLES))
TIME_LINES = dict(zip((3, 6, 9), SERIES_STYLES))
TIME_MARKERS = dict(zip((3, 6, 9), ('o', 's', '^')))
LIQUID_MARKERS = dict(zip(LIQUID_COLORS, ('o', 's', '^', 'D')))
ADHESION_MEAN_HALF_LENGTH = .20  # half-width of each mean bar
ADHESION_METHOD_OFFSETS = {'Blister': -ADHESION_MEAN_HALF_LENGTH,
                           'Peeling': ADHESION_MEAN_HALF_LENGTH}
ADHESION_MEAN_COLORS = MEAN_COLORS
ADHESION_MEAN_EDGE_COLOR = '#A6A6A6'
ADHESION_MEAN_EDGE_WIDTH = .70  # pt; quiet outlines for both gray mean bars


def setup_publication_style():
    os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
    for name in ('arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf',
                 'times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        path = Path('C:/Windows/Fonts') / name
        if path.exists():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        'font.family': HOUSE_FONT, 'font.size': 9.5, 'axes.labelsize': HOUSE_AXIS_LABEL,
        'axes.titlesize': HOUSE_CONDITION, 'xtick.labelsize': HOUSE_TICK_LABEL, 'ytick.labelsize': HOUSE_TICK_LABEL,
        'legend.fontsize': HOUSE_LEGEND, 'legend.title_fontsize': HOUSE_CONDITION,
        'mathtext.fontset': 'stix', 'axes.linewidth': HOUSE_FRAME,
        'axes.spines.top': True, 'axes.spines.right': True,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'axes.edgecolor': INK,
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
        'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'svg.hashsalt':'adhesion-submission', 'svg.hashsalt': 'asme-paper-figures', 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'path.simplify': False,
        'legend.frameon': True, 'legend.fancybox': False, 'legend.framealpha': 1,
        'legend.facecolor': 'white', 'legend.edgecolor': INK, 'savefig.dpi': 900,
    })


def apply_axis_style(ax):
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(HOUSE_FRAME); spine.set_color(INK)
    ax.tick_params(direction='in', length=3.5, width=.8, pad=4,
                   top=True, right=True)
    ax.grid(False)


def add_panel_label(ax, letter, title=None, condition=None):
    # Fixed point offsets align headers for unequal-height axes (Fig. 3).
    height = 23 if condition else 8
    ax.annotate(f'({letter.strip("()")})', xy=(0, 1), xycoords='axes fraction',
                xytext=(0, height), textcoords='offset points',
                fontsize=HOUSE_PANEL, fontweight='bold', va='bottom', annotation_clip=False)
    if title:
        ax.annotate(title, xy=(0, 1), xycoords='axes fraction',
                    xytext=(21, height), textcoords='offset points',
                    fontsize=HOUSE_CONDITION, va='bottom', annotation_clip=False)
    if condition:
        ax.annotate(condition, xy=(0, 1), xycoords='axes fraction',
                    xytext=(0, 7), textcoords='offset points',
                    fontsize=HOUSE_CONDITION, color=INK, va='bottom', annotation_clip=False)


def _opaque(rgba, alpha):
    a = colors.to_rgba_array(rgba)
    if len(a):
        weight = a[:, 3:4] if alpha is None else float(alpha)
        a[:, :3] = a[:, :3]*weight + (1-weight)
        a[:, 3] = 1
    return a


def save_publication_figure(fig, destination, composite=False):
    """Save the manuscript vector PDF; temporary previews are not submission files."""
    destination = Path(destination)
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            # Maintain the 6 pt floor when these figures are placed at the
            # manuscript's 173.7 mm text width (rather than their tight crop).
            if text.get_fontsize() < 7:
                text.set_fontsize(7)
            # Mathtext shrinks scripts to 70%; preserve the 6 pt glyph floor
            # after placement at the current manuscript text width.
            if '$' in text.get_text() and any(c in text.get_text() for c in ('_', '^')):
                text.set_fontsize(max(text.get_fontsize(), 9.2))
            assert text.get_fontsize() >= 6, text.get_text()
    fig.canvas.draw()
    kw = dict(bbox_inches='tight', pad_inches=.045, facecolor='white',
              transparent=False, dpi=500 if composite else 900)
    fig.savefig(destination.with_suffix('.pdf'),
                metadata={'CreationDate': None, 'ModDate': None}, **kw)

def adhesion_figure(points, means, ylim=(0, 2.15), insets=None):
    """Use filled circles and left/right gray bars from zero to each method mean."""
    liquids = tuple(LIQUID_COLORS)
    layout=json.loads((CONFIG/'figure_layout.json').read_text(encoding='utf-8'))['figure4']
    fig, axes = plt.subplots(1, 4, sharey=False, figsize=layout['figsize_inches'])
    fig.subplots_adjust(**layout['margins'])
    plotted = 0
    for ax, liquid, letter in zip(axes, liquids, 'abcd'):
        ax.set(xlim=(-.48, 2.48), ylim=ylim, xticks=(0, 1, 2), xticklabels=('3', '6', '9'))
        apply_axis_style(ax)
        ax.spines[['top', 'right', 'left']].set_visible(True)
        ax.tick_params(axis='y', labelsize=HOUSE_TICK_LABEL)
        ax.set_xlabel('Conditioning time (h)',fontsize=HOUSE_AXIS_LABEL)
        ax.set_ylabel(formatted(r'Apparent adhesion, \(\Gamma\) (J/m\(^{2}\))'),labelpad=3.853693)
        ax.set_yticks([0,.5,1,1.5,2])
        ax.annotate('('+letter+')', xy=(0, 1), xycoords='axes fraction',
                    xytext=layout['panel_offset_pt'], textcoords='offset points', fontsize=HOUSE_PANEL,
                    va='bottom', annotation_clip=False)
        ax.annotate(liquid, xy=(.5, 1), xycoords='axes fraction', xytext=(0, -4),
                    textcoords='offset points', fontsize=HOUSE_CONDITION, ha='center',
                    va='top', annotation_clip=False)
        local = points[points.liquid.eq(liquid)]
        for i, time in enumerate((3, 6, 9)):
            row = means[means.liquid.eq(liquid) & means.nominal_time_h.eq(time)].iloc[0]
            for method, offset in ADHESION_METHOD_OFFSETS.items():
                subset = local[local.method.eq(method) & local.display_nominal_group_h.eq(time)]
                count_field = 'number_of_blister_records' if method == 'Blister' else 'number_of_peeling_records'
                mean_field = 'mean_Gamma_B_J_m2' if method == 'Blister' else 'mean_Gamma_P_J_m2'
                assert len(subset) == int(row[count_field])
                if not len(subset):
                    continue
                value = row[mean_field]
                assert np.isclose(subset.Gamma_J_m2.mean(), value, rtol=1e-12)
                center = i+offset
                ax.add_patch(Rectangle((center-ADHESION_MEAN_HALF_LENGTH, 0),
                                       2*ADHESION_MEAN_HALF_LENGTH, value,
                                       facecolor=ADHESION_MEAN_COLORS[method],
                                       edgecolor=ADHESION_MEAN_EDGE_COLOR,
                                       linewidth=ADHESION_MEAN_EDGE_WIDTH, zorder=2))
        for method in ADHESION_METHOD_OFFSETS:
            for h in ADHESION_THICKNESS_COLORS:
                data = local[local.method.eq(method) & local.thickness_um.eq(h)]
                xx = (data.display_nominal_group_h-3)/3 + data.display_offset
                ax.scatter(xx, data.Gamma_J_m2, marker='o', s=24,
                           facecolors=ADHESION_THICKNESS_COLORS[h],
                           edgecolors=ADHESION_THICKNESS_COLORS[h], linewidths=.8, zorder=4)
                for x, (_, row) in zip(xx, data.iterrows()):
                    if row.protocol_differs:
                        ax.annotate('†', (x, row.Gamma_J_m2), xytext=(2, 3),
                                    textcoords='offset points', fontsize=6.5)
                plotted += len(data)
    assert plotted == len(points)
    thickness = [Line2D([], [], color=ADHESION_THICKNESS_COLORS[h], marker='o', lw=0,
                        markersize=4, markeredgewidth=.8, label=rf'{h} $\mu$m')
                 for h in ADHESION_THICKNESS_COLORS]
    methods = [Patch(facecolor=ADHESION_MEAN_COLORS[method],
                     edgecolor=ADHESION_MEAN_EDGE_COLOR,
                     linewidth=ADHESION_MEAN_EDGE_WIDTH,
                     label=formatted(label))
               for method, label in (
                   ('Blister', r'Blister mean, \(\Gamma_B\)'),
                   ('Peeling', r'Peeling mean, \(\Gamma_P\)'))]
    legends=[]
    for handles,columns in [(thickness,4),(methods,2)]:
        legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,layout['legend_top_anchor_y']),
            ncol=columns,handletextpad=.35,columnspacing=1.15,borderpad=.35)
        legend.get_frame().set_linewidth(HOUSE_LEGEND_FRAME)
        legends.append(legend)
    fig.canvas.draw()
    widths=[legend.get_window_extent(fig.canvas.get_renderer()).width/fig.bbox.width for legend in legends]
    center=layout['legend_group_center_x'];gap=layout['legend_gap_fraction']
    for legend,x in zip(legends,[center-(widths[1]+gap)/2,center+(widths[0]+gap)/2]):
        legend.set_bbox_to_anchor((x,layout['legend_top_anchor_y']),transform=fig.transFigure)
    if insets:
        for image, elapsed, left in zip(insets, (10, 15), (.05, .53)):
            # Reserve the same clear top strip for the liquid title as in (a-c).
            inset=axes[3].inset_axes([left, .55, .42, .34])
            inset.imshow(image, interpolation='none', origin='lower')
            inset.set(xticks=[], yticks=[])
            for spine in inset.spines.values():
                spine.set_linewidth(.8); spine.set_edgecolor(INK)
            inset.text(.5, .035, f'{elapsed} s', transform=inset.transAxes,
                       ha='center', va='bottom', fontsize=8.5, color='white',
                       bbox={'facecolor':'black','edgecolor':'none','alpha':.65,'pad':1})
        axes[3].text(.5, .53, r'150 $\mu$m, 9 h', transform=axes[3].transAxes,
                     ha='center', va='top', fontsize=8.5)
    return fig
