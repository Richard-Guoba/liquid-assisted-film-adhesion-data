"""Compatibility helpers for the finalized, simplified peeling metadata."""

import numpy as np


def apply_peeling_configuration(results, config):
    """Apply author-confirmed conditions by unique raw-sheet ID.

    Existing mechanical estimates are retained. A changed width or plateau
    interval requires recalculation rather than relabelling a cached estimate.
    Neither filenames nor force magnitudes are used to assign conditions.
    """
    required = {
        'test_id', 'sheet_name', 'liquid', 'thickness_um', 'drying_time_h',
        'specimen_width_mm', 'plateau_start_mm_zeroed',
        'plateau_end_mm_zeroed',
    }
    for label, frame in [('configuration', config), ('results', results)]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'Peeling {label} is missing columns: {sorted(missing)}')
        for key in ['sheet_name', 'test_id']:
            if frame[key].isna().any() or frame[key].duplicated().any():
                raise ValueError(f'Peeling {label} requires unique, nonmissing {key}')
    if set(results.sheet_name) != set(config.sheet_name):
        raise ValueError('Peeling results/configuration raw-sheet IDs differ')

    aligned = config.set_index('sheet_name').loc[results.sheet_name].copy()
    aligned.index = results.index
    for key in ['test_id', 'specimen_width_mm', 'plateau_start_mm_zeroed',
                'plateau_end_mm_zeroed']:
        same = np.isclose(results[key].to_numpy(dtype=float),
                          aligned[key].to_numpy(dtype=float),
                          rtol=1e-12, atol=1e-12, equal_nan=False)
        if not same.all():
            ids = results.loc[~same, 'sheet_name'].tolist()
            raise ValueError(f'Peeling cached {key} differs from configuration: {ids}')

    corrected = results.copy(deep=True)
    for key in ['liquid', 'thickness_um', 'drying_time_h']:
        corrected[key] = aligned[key]
    if 'planned_drying_time_h' in corrected:
        corrected['planned_drying_time_h'] = (
            aligned['planned_drying_time_h'] if 'planned_drying_time_h' in aligned
            else aligned['drying_time_h']
        )
    return corrected


def normalize_peeling_config(frame):
    """Return a copy with metadata columns used by the unchanged figure code."""
    normalized = frame.copy(deep=True)
    normalized['liquid'] = normalized['liquid'].str.lower().replace({'oil': 'glycerol'})
    if 'planned_drying_time_h' not in normalized:
        normalized['planned_drying_time_h'] = normalized['drying_time_h']
    if 'peel_angle_deg' not in normalized:
        normalized['peel_angle_deg'] = np.nan
    return normalized


def figure2_label(config, sheet_name, mode):
    """Build a Figure 2 legend label from the finalized configuration row."""
    row = config.loc[config.sheet_name.eq(sheet_name)]
    if len(row) != 1:
        raise ValueError(f'Expected one configuration row for {sheet_name}, found {len(row)}.')
    row = row.iloc[0]
    time_h = f'{float(row.drying_time_h):g}'
    if mode == 'time_speed':
        speed = f'{float(row.peel_speed_mm_per_min):g}'
        return f'{time_h} h / {speed} mm min$^{{-1}}$'
    if mode == 'liquid_time':
        liquid = {'ethanol': 'Ethanol', 'ipa': 'IPA', 'water': 'Water',
                  'oil': 'Glycerol', 'glycerol': 'Glycerol'}[str(row.liquid).lower()]
        return f'{liquid} / {time_h} h'
    raise ValueError(f'Unknown Figure 2 label mode: {mode}')


def build_blister_mapping(index, force):
    """Rebuild the legacy cache contract from finalized result workbooks."""
    force_key = force.assign(
        sample=lambda d: (
            d.liquid.str.lower().replace({'glycerol': 'oil'}) + '_'
            + d.thickness_um.astype(int).astype(str) + '_'
            + d.ideal_dry_time_h.astype(int).astype(str)
            + d.repeat_id.astype(int).map(lambda n: '' if n == 0 else f'_{n}')
        ),
        force_test_id=lambda d: d.test_id,
        force_slope_n_per_mm=lambda d: d.slope_n_per_mm,
        intercept_n=lambda d: d.fit_intercept_n,
    )
    # The finalized workbooks contain all configured records but no supported
    # primary/excluded classification, so every source record is retained equally.
    force_key['source_primary'] = True

    index_mapping = index.iloc[:, [0, 2, 3, 4, 5, 7]].copy()
    index_mapping.columns = [
        'sample', 'thickness_um', 'time_h', 'force_slope_n_per_mm_index',
        'radius_slope_mm_per_mm', 'E_MPa',
    ]
    mapping = force_key[[
        'sample', 'force_test_id', 'repeat_id', 'force_slope_n_per_mm',
        'intercept_n', 'source_primary',
    ]].merge(index_mapping, on='sample', validate='one_to_one')
    if not np.allclose(mapping.force_slope_n_per_mm,
                       mapping.force_slope_n_per_mm_index, rtol=1e-12, atol=1e-12):
        raise ValueError('Force slopes disagree between finalized index and fit results.')
    return mapping.drop(columns='force_slope_n_per_mm_index').set_index('sample')
