"""
Copyright (C) 2025 Beihang University, Neuromorphic Vision Perception and Computing Group

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright © Beihang University, Neuromorphic Vision Perception and Computing Group.
License: This code is licensed under the GNU General Public License v3.0.
You can redistribute it and/or modify it under the terms of the GPL-3.0 License.
"""


from __future__ import annotations

import argparse
import csv
import io
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False

EPS = 1e-12


# ---------------- data ---------------- #

@dataclass
class FileMetric:
    name: str = ""
    event_count: int = 0
    structural_aocc: float = 0.0

    # per-tile FFT outputs (diagnostic only)
    flicker_purity_per_tile: np.ndarray = None
    narrow_band_energy_per_tile: np.ndarray = None
    total_band_energy_per_tile: np.ndarray = None
    tile_event_counts: np.ndarray = None
    rel_penalty: float = 0.0
    detected_peaks_hz: list = None
    detected_freq_hz: float = 0.0

    # NEW headline residual: peak-vs-background within sequence's own
    # tile-aggregated spectrum, using top-N peaks identified on raw.
    residual_ratio: float = 0.0          # compressed residual used for score; raw = 1 when raw peak exists
    residual_linear_ratio: float = 0.0   # raw-normalized linear peak excess ratio before power compression
    effective_residual_ratio: float = 0.0
    band_residual_ratio: float = 0.0      # diagnostic: peak_excess / total_band within this method
    raw_peak_excess_for_norm: float = 0.0      # raw linear peak excess, diagnostic only
    db_peak_prominence: float = 0.0          # dB peak-over-background prominence used for residual
    raw_db_peak_prominence_for_norm: float = 0.0
    score: float = 0.0
    peak_excess_energy: float = 0.0          # linear peak-minus-background excess, diagnostic only
    total_band_energy_in_mask: float = 0.0
    background_median: float = 0.0
    event_ratio_in_mask: float = 0.0          # diagnostic only

    # bookkeeping
    group_id: str = ""
    method: str = ""
    source_path: str = ""
    spatial_mask_source: str = ""
    n_flicker_tiles: int = 0
    residual_peak_freqs_hz: list = None

    # transient buffers used by plotting and the residual computation
    dft_freqs_hz: np.ndarray = None
    dft_power_db: np.ndarray = None
    aggregate_freqs_hz: np.ndarray = None
    aggregate_power: np.ndarray = None
    n_active_tiles: int = 0
    band_power_full: np.ndarray = None
    band_freqs_full: np.ndarray = None
    active_tile_mask: np.ndarray = None
    peak_mask_full: np.ndarray = None


# ---------------- utility ---------------- #

def parse_four_numeric_tokens(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.replace(",", " ").split()
    if len(parts) < 4:
        return None
    try:
        vals = [float(parts[i]) for i in range(4)]
    except (ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    return vals


def tail_lines(path: Path, max_lines: int = 256, block_size: int = 65536) -> List[str]:
    if max_lines <= 0:
        return []
    data = b""
    with path.open("rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        while pos > 0 and data.count(b"\n") <= max_lines:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            data = f.read(read_size) + data
    lines = data.splitlines()[-max_lines:]
    return [line.decode("utf-8", errors="ignore") for line in lines]


def detect_event_format(path: Path, tail_sample_lines: int = 2000,
                        first_col_time_threshold: float = 1100.0) -> str:
    first_cols: List[float] = []
    for line in tail_lines(path, max_lines=tail_sample_lines):
        v = parse_four_numeric_tokens(line)
        if v is None:
            continue
        first_cols.append(v[0])
    if not first_cols:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                v = parse_four_numeric_tokens(line)
                if v is None:
                    continue
                first_cols.append(v[0])
                if len(first_cols) >= min(64, tail_sample_lines):
                    break
    if not first_cols:
        raise ValueError(f"Cannot detect event format for {path}")
    avg = float(np.mean(np.asarray(first_cols, dtype=np.float64)))
    fmt = "txyp" if avg > first_col_time_threshold else "xypt"
    print(f"  format auto: {path.name} avg_first_col_tail={avg:.2f} -> {fmt}")
    return fmt


def trapezoid_area(xs, ys, x_min: float, x_max: float) -> float:
    pairs = sorted((x, y) for x, y in zip(xs, ys) if x_min <= x <= x_max)
    if len(pairs) < 2:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
        area += (x1 - x0) * (y0 + y1) * 0.5
    return float(area)


# ---------------- fast event reader ---------------- #

def fast_read_events(path: Path, input_format: str = "auto",
                      max_events: int = 0,
                      tail_sample_lines: int = 256,
                      first_col_time_threshold: float = 10000.0,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    fmt = detect_event_format(path, tail_sample_lines, first_col_time_threshold) \
          if input_format == "auto" else input_format

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
    except OSError as e:
        raise IOError(f"failed to read {path}: {e}")
    if "," in raw_text:
        raw_text = raw_text.replace(",", " ")
    text_buf = io.StringIO(raw_text)

    arr = None
    if HAVE_PANDAS:
        for kwargs in (
            {"sep": r"\s+", "engine": "c"},
            {"delim_whitespace": True, "engine": "c"},
            {"sep": r"\s+", "engine": "python"},
        ):
            try:
                text_buf.seek(0)
                df = pd.read_csv(
                    text_buf, header=None, comment='#',
                    usecols=[0, 1, 2, 3], dtype=np.float64,
                    on_bad_lines='skip', **kwargs,
                )
                arr = df.to_numpy()
                break
            except (TypeError, ValueError) as e:
                last_err = e
                continue
        if arr is None:
            print(f"  pd.read_csv all variants failed ({last_err}); slow path")

    if arr is None:
        rows = []
        text_buf.seek(0)
        for line in text_buf:
            v = parse_four_numeric_tokens(line)
            if v is not None:
                rows.append(v)
        if not rows:
            return (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), fmt)
        arr = np.asarray(rows, dtype=np.float64)

    if arr.size == 0:
        return (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), fmt)

    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]

    if fmt == "txyp":
        t = arr[:, 0]; x = arr[:, 1]; y = arr[:, 2]; p = arr[:, 3]
    else:
        x = arr[:, 0]; y = arr[:, 1]; p = arr[:, 2]; t = arr[:, 3]

    order = np.argsort(t, kind='stable')
    t = t[order]
    x = x[order].astype(np.int32, copy=False)
    y = y[order].astype(np.int32, copy=False)
    p = p[order].astype(np.int32, copy=False)

    if t.size > 0:
        t = t - t[0]
    if max_events > 0 and t.size > max_events:
        t = t[:max_events]; x = x[:max_events]; y = y[:max_events]; p = p[:max_events]

    return t, x, y, p, fmt


# ---------------- fast tile signal builder ---------------- #

def fast_build_tile_signals(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                              W: int, H: int, microbin_us: float,
                              tile_cols: int, tile_rows: int,
                              ) -> Tuple[np.ndarray, np.ndarray]:
    n_tiles = max(1, tile_cols * tile_rows)
    if t.size == 0:
        return np.zeros((n_tiles, 16), dtype=np.float64), np.zeros(n_tiles, dtype=np.float64)

    duration_us = max(float(t[-1] - t[0]), float(microbin_us))
    n_bins = max(16, int(np.ceil(duration_us / microbin_us)) + 1)

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not valid.any():
        return np.zeros((n_tiles, n_bins), dtype=np.float64), np.zeros(n_tiles, dtype=np.float64)
    t_v = t[valid]; x_v = x[valid]; y_v = y[valid]

    bin_idx = np.floor((t_v - t_v[0]) / microbin_us).astype(np.int64)
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)
    if not in_range.any():
        return np.zeros((n_tiles, n_bins), dtype=np.float64), np.zeros(n_tiles, dtype=np.float64)
    bin_idx = bin_idx[in_range]
    x_v = x_v[in_range]; y_v = y_v[in_range]

    tx = np.minimum(tile_cols - 1, np.maximum(0, (x_v.astype(np.int64) * tile_cols) // W))
    ty = np.minimum(tile_rows - 1, np.maximum(0, (y_v.astype(np.int64) * tile_rows) // H))
    tile_idx = (ty * tile_cols + tx).astype(np.int64)

    signals = np.zeros((n_tiles, n_bins), dtype=np.float64)
    np.add.at(signals, (tile_idx, bin_idx), 1.0)
    tile_counts = np.bincount(tile_idx, minlength=n_tiles).astype(np.float64)
    return signals, tile_counts


# ---------------- per-tile FFT pass (diagnostic) ---------------- #

def compute_flicker_purity(
    signals: np.ndarray,
    tile_counts: np.ndarray,
    microbin_us: float,
    hp_cutoff_hz: float,
    periodic_fmax_hz: float,
    peak_ratio_threshold: float,
    diagnostic_top_k: int = 5,
    min_active_energy: float = 1e-8,
    min_events_per_tile: int = 0,
    peak_mask_override: Optional[np.ndarray] = None,
    peak_dilation_bins: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, float, float,
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-tile FFT and per-tile purity. Used for diagnostic heatmaps and
    to fill band_power_full, which the headline residual function consumes.
    The headline residual itself is computed in
    compute_peak_vs_background_residual on the tile-aggregated spectrum.
    """
    n_tiles, n_bins = signals.shape
    purity = np.zeros(n_tiles, dtype=np.float64)
    narrow_e = np.zeros(n_tiles, dtype=np.float64)
    total_e = np.zeros(n_tiles, dtype=np.float64)
    active_mask = np.zeros(n_tiles, dtype=bool)
    empty_band = np.zeros(0, dtype=np.float64)
    empty_2d = np.zeros((0, 0), dtype=np.float64)

    if n_bins < 16:
        return (purity, narrow_e, total_e, [], 0.0, 0.0,
                empty_band, empty_band, active_mask, empty_2d,
                np.zeros((n_tiles, 0), dtype=bool))

    sig = signals - signals.mean(axis=1, keepdims=True)
    time_energy = (sig * sig).sum(axis=1)

    gate_energy = time_energy > min_active_energy
    if min_events_per_tile > 0:
        gate_events = tile_counts >= float(min_events_per_tile)
    else:
        gate_events = np.ones(n_tiles, dtype=bool)
    active = gate_energy & gate_events
    if not active.any():
        return (purity, narrow_e, total_e, [], 0.0, 0.0,
                empty_band, empty_band, active_mask, empty_2d,
                np.zeros((n_tiles, 0), dtype=bool))

    dt_s = microbin_us * 1e-6
    freqs = np.fft.rfftfreq(n_bins, d=dt_s)
    nyquist = 0.5 / dt_s
    fmax = min(float(periodic_fmax_hz), 0.95 * nyquist)
    band_mask = (freqs >= hp_cutoff_hz) & (freqs <= fmax)
    band_freqs = freqs[band_mask]
    n_band = band_freqs.size
    if n_band < 3:
        return (purity, narrow_e, total_e, [], 0.0, 0.0,
                empty_band, empty_band, active_mask, empty_2d,
                np.zeros((n_tiles, 0), dtype=bool))

    spec = np.fft.rfft(sig[active], axis=1)
    power = (spec.real ** 2 + spec.imag ** 2)
    band_power = power[:, band_mask]

    if peak_mask_override is not None:
        raw_band_freqs, raw_peak_mask = peak_mask_override
        if raw_peak_mask.shape[0] != n_tiles:
            raise ValueError(
                f"peak_mask_override has {raw_peak_mask.shape[0]} tile "
                f"rows, expected {n_tiles}.")
        nn_idx = np.argmin(
            np.abs(raw_band_freqs[np.newaxis, :] - band_freqs[:, np.newaxis]),
            axis=1)
        rebuilt = raw_peak_mask[:, nn_idx]
        is_peak = rebuilt[active]
    else:
        background_for_detection = np.median(band_power, axis=1,
                                                keepdims=True) + EPS
        is_peak = band_power > (peak_ratio_threshold * background_for_detection)

    if peak_dilation_bins > 0:
        try:
            from scipy.ndimage import binary_dilation
            structure = np.ones(2 * peak_dilation_bins + 1, dtype=bool)
            is_peak = binary_dilation(is_peak, structure=structure[np.newaxis, :])
        except ImportError:
            dilated = is_peak.copy()
            for shift in range(1, peak_dilation_bins + 1):
                dilated[:, shift:] |= is_peak[:, :-shift]
                dilated[:, :-shift] |= is_peak[:, shift:]
            is_peak = dilated

    non_peak = ~is_peak
    n_nonpeak_per_row = non_peak.sum(axis=1, keepdims=True)
    background_per_tile = np.zeros((band_power.shape[0], 1), dtype=np.float64)
    has_nonpeak = (n_nonpeak_per_row.flatten() > 0)
    if has_nonpeak.any():
        bp_masked = np.where(non_peak, band_power, np.nan)
        with np.errstate(invalid='ignore'):
            background_per_tile[has_nonpeak, 0] = np.nanmedian(
                bp_masked[has_nonpeak], axis=1)
    if (~has_nonpeak).any():
        background_per_tile[~has_nonpeak, 0] = np.median(
            band_power[~has_nonpeak], axis=1)

    peak_diff = np.maximum(0.0, band_power - background_per_tile)
    narrow_sum = (peak_diff * is_peak).sum(axis=1)
    total_sum = band_power.sum(axis=1) + EPS
    purity_active = narrow_sum / total_sum

    active_idx = np.where(active)[0]
    purity[active_idx] = purity_active
    narrow_e[active_idx] = narrow_sum
    total_e[active_idx] = total_sum - EPS
    active_mask[active_idx] = True

    aggregate_full = band_power.sum(axis=0)
    aggregate_peak = (peak_diff * is_peak).sum(axis=0)

    detected_peaks_hz: list = []
    if aggregate_peak.max() > 0:
        order = np.argsort(aggregate_peak)[::-1][:diagnostic_top_k]
        for j in order:
            if aggregate_peak[j] > 0:
                detected_peaks_hz.append((float(band_freqs[j]),
                                            float(aggregate_peak[j])))

    n_act = active_idx.size
    n_top = max(1, int(np.ceil(0.2 * n_act)))
    rel_penalty = float(np.mean(np.partition(purity_active, max(0, n_act - n_top))[n_act - n_top:])) \
                   if n_act > 0 else 0.0
    detected_freq_hz = detected_peaks_hz[0][0] if detected_peaks_hz else 0.0

    band_power_full = np.zeros((n_tiles, n_band), dtype=np.float64)
    band_power_full[active_idx] = band_power
    peak_mask_full = np.zeros((n_tiles, n_band), dtype=bool)
    peak_mask_full[active_idx] = is_peak

    return (purity, narrow_e, total_e, detected_peaks_hz, rel_penalty,
            detected_freq_hz, band_freqs, aggregate_full, active_mask,
            band_power_full, peak_mask_full)


# ---------------- structural AOCC ---------------- #

def sobel_structural_contrast_fast(frame: np.ndarray, beta: float,
                                     gaussian_kernel: int = 5,
                                     gaussian_sigma: float = 2.0) -> float:
    mass = float(frame.sum())
    if mass <= EPS:
        return 0.0
    z = np.log1p(beta * frame / (mass + EPS)).astype(np.float64, copy=False)
    if gaussian_kernel > 1:
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        z = cv2.GaussianBlur(z, (gaussian_kernel, gaussian_kernel), gaussian_sigma)
    grad_x = cv2.Sobel(z, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(z, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    return float(np.std(mag, ddof=1)) if mag.size > 1 else 0.0


def fast_structural_aocc(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                          W: int, H: int,
                          min_interval_us: float, max_interval_us: float,
                          step_us: float, aocc_x_min: float, aocc_x_max: float,
                          beta: float, ccc_path: Path,
                          max_windows_per_interval: int = 0,
                          contrast_interval_stride: int = 1,
                          ) -> float:
    if t.size == 0:
        return 0.0

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not valid.any():
        return 0.0
    t_v = t[valid]; x_v = x[valid]; y_v = y[valid]
    t0 = float(t_v[0])

    intervals_us = np.arange(min_interval_us, max_interval_us, step_us, dtype=np.float64)
    if contrast_interval_stride > 1:
        intervals_us = intervals_us[::contrast_interval_stride]

    ccc_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [["Interval_us", "Mean_Structural_Contrast",
             "Median_Structural_Contrast", "RMS_Structural_Contrast"]]

    interval_xs: List[float] = []
    interval_ys: List[float] = []

    n_int = len(intervals_us)
    for k, interval_us in enumerate(intervals_us):
        bin_idx = np.floor((t_v - t0) / interval_us).astype(np.int64)
        if bin_idx.size == 0:
            mean_c = median_c = rms_c = 0.0
        else:
            change = np.diff(bin_idx) != 0
            boundaries = np.concatenate(([0], np.where(change)[0] + 1, [bin_idx.size]))
            contrasts: List[float] = []
            used = 0
            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                if e == s:
                    continue
                frame = np.zeros((H, W), dtype=np.float64)
                np.add.at(frame, (y_v[s:e], x_v[s:e]), 1.0)
                if frame.sum() <= 0:
                    continue
                contrasts.append(sobel_structural_contrast_fast(frame, beta))
                used += 1
                if max_windows_per_interval > 0 and used >= max_windows_per_interval:
                    break
            if contrasts:
                arr = np.asarray(contrasts, dtype=np.float64)
                mean_c = float(arr.mean())
                median_c = float(np.median(arr))
                rms_c = float(np.sqrt((arr * arr).mean()))
            else:
                mean_c = median_c = rms_c = 0.0

        rows.append([f"{interval_us:.6f}", f"{mean_c:.12f}",
                     f"{median_c:.12f}", f"{rms_c:.12f}"])
        interval_xs.append(float(interval_us))
        interval_ys.append(mean_c)
        print(f"\r  AOCC interval {k+1}/{n_int}: {interval_us:.0f}us", end='', flush=True)
    print()

    with ccc_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    return trapezoid_area(interval_xs, interval_ys, aocc_x_min, aocc_x_max)


# ---------------- per-file processing ---------------- #

def process_single_path(args, txt_path: Path, group_id: str, method: str,
                          compute_dft: bool = False,
                          peak_mask_override: Optional[np.ndarray] = None
                          ) -> FileMetric:
    ccc_dir = Path(args.ccc_dir)
    aocc_x_max = args.max_interval_us - 1.0 if args.aocc_x_max < 0 else args.aocc_x_max
    beta = float(args.width * args.height) if args.beta <= 0 else float(args.beta)

    print(f"Processing group={group_id} method={method} file={txt_path.name}")
    t_start = time.time()
    t_arr, x_arr, y_arr, p_arr, fmt = fast_read_events(
        txt_path,
        input_format=args.input_format,
        max_events=getattr(args, "max_events", 0),
        tail_sample_lines=args.format_tail_lines,
        first_col_time_threshold=args.first_col_time_threshold,
    )
    t_read = time.time() - t_start
    print(f"  read {t_arr.size} events in {t_read:.1f}s "
          f"({t_arr.size/max(t_read, 1e-3):.0f} ev/s)")
    if t_arr.size > 0:
        print(f"  first_t={t_arr[0]:.0f}us, last_t={t_arr[-1]:.0f}us")

    base = txt_path.stem
    safe_base = f"group_{group_id}_{method}_{base}".replace("/", "_").replace(":", "_")

    if getattr(args, "skip_structural_aocc", False):
        structural_aocc = 0.0
    else:
        t_aocc = time.time()
        structural_aocc = fast_structural_aocc(
            t=t_arr, x=x_arr, y=y_arr,
            W=args.width, H=args.height,
            min_interval_us=args.min_interval_us,
            max_interval_us=args.max_interval_us,
            step_us=args.step_us,
            aocc_x_min=args.aocc_x_min,
            aocc_x_max=aocc_x_max,
            beta=beta,
            ccc_path=ccc_dir / f"{safe_base}_ccc.csv",
            max_windows_per_interval=getattr(args, "max_windows_per_interval", 0),
            contrast_interval_stride=max(1, getattr(args, "contrast_interval_stride", 1)),
        )
        print(f"  AOCC done in {time.time() - t_aocc:.1f}s -> {structural_aocc:.4f}")

    t_fft = time.time()
    signals, tile_counts = fast_build_tile_signals(
        t_arr, x_arr, y_arr,
        W=args.width, H=args.height,
        microbin_us=args.microbin_us,
        tile_cols=args.tile_cols, tile_rows=args.tile_rows,
    )
    (purity, narrow_e, total_e, peaks, rel_pen, det_freq,
     band_freqs, agg_power, active_mask,
     band_power_full, peak_mask_full) = compute_flicker_purity(
        signals=signals, tile_counts=tile_counts,
        microbin_us=args.microbin_us,
        hp_cutoff_hz=args.hp_cutoff_hz, periodic_fmax_hz=args.periodic_fmax_hz,
        peak_ratio_threshold=args.peak_ratio_threshold,
        diagnostic_top_k=args.diagnostic_top_k,
        min_events_per_tile=int(args.min_events_per_tile),
        peak_mask_override=peak_mask_override,
        peak_dilation_bins=int(args.peak_dilation_bins),
    )
    n_active = int(active_mask.sum())
    n_tiles_total = int(active_mask.size)
    print(f"  FFT done in {time.time() - t_fft:.1f}s | "
          f"active tiles {n_active}/{n_tiles_total} | "
          f"rel_penalty={rel_pen:.4f} "
          f"detected_top_peaks_hz={[f'{f:.1f}' for f, _ in peaks[:3]]}")

    dft_freqs = None
    dft_power = None
    if compute_dft:
        spectrum = compute_global_dft(
            t=t_arr, x=x_arr, y=y_arr,
            W=args.width, H=args.height,
            microbin_us=args.microbin_us,
            fmax_hz=float(args.dft_fmax_hz),
            hp_cutoff_hz=0.0,
        )
        if spectrum is not None:
            dft_freqs, dft_power = spectrum

    m = FileMetric(
        name=base,
        event_count=int(t_arr.size),
        structural_aocc=structural_aocc,
        flicker_purity_per_tile=purity,
        narrow_band_energy_per_tile=narrow_e,
        total_band_energy_per_tile=total_e,
        tile_event_counts=tile_counts,
        rel_penalty=rel_pen,
        detected_peaks_hz=peaks,
        detected_freq_hz=det_freq,
    )
    m.group_id = group_id
    m.method = method
    m.source_path = str(txt_path)
    m.dft_freqs_hz = dft_freqs
    m.dft_power_db = dft_power
    m.aggregate_freqs_hz = band_freqs
    m.aggregate_power = agg_power
    m.n_active_tiles = n_active
    m.band_power_full = band_power_full
    m.band_freqs_full = band_freqs
    m.active_tile_mask = active_mask
    m.peak_mask_full = peak_mask_full
    return m


# ---------------- residual: peak-vs-background within sequence ---------------- #

def compute_peak_vs_background_residual(
        metrics: List[FileMetric],
        spatial_mask: np.ndarray,
        lambda_value: float,
        n_residual_peaks: int = 3,
        peak_half_width_bins: int = 2,
        detection_threshold_factor: float = 2.0,
        score_mode: str = "raw_aocc_exp",
        aocc_alpha: float = 0.5,
        residual_gain: float = 3.0,
        penalty_power: float = 1.0,
        residual_zero_threshold: float = 0.0,
        peak_excess_bg_threshold_factor: float = 0.0,
        peak_db_threshold_db: float = 0.0,
        residual_power: float = 1.0,
) -> None:
    """Headline flicker ratio using the original raw-normalized framework.

    This version keeps the old convention:
        residual(raw) = 1.0
    when a valid raw flicker peak exists.

    The flicker strength used for the headline residual is computed in the dB
    domain so it matches the visual interpretation of the spectrum plot:

        db_prominence(m, f) = 10*log10((agg_m(f)+eps)/(bg_median(m)+eps))
        db_excess(m, f)     = max(0, db_prominence(m, f) - delta_db)
        F_db(m)             = mean_{f in peak bins} db_excess(m, f)
        linear_ratio(m)     = F_db(m) / F_db(raw)
        residual(m)         = linear_ratio(m) ** residual_power

    The linear power-domain peak-minus-background excess is still computed and
    recorded as a diagnostic value, but it is NOT the headline residual.  This
    avoids the huge dynamic range caused by unnormalized FFT power while still
    measuring whether flicker peaks stand above the local background.
    """
    raw = metrics[0]
    band_freqs = raw.band_freqs_full
    raw_aocc_ref = max(float(raw.structural_aocc), EPS)
    residual_zero_threshold = max(0.0, float(residual_zero_threshold))
    peak_excess_bg_threshold_factor = max(0.0, float(peak_excess_bg_threshold_factor))
    peak_db_threshold_db = max(0.0, float(peak_db_threshold_db))
    residual_power = max(float(residual_power), EPS)

    def _effective_residual(residual: float) -> float:
        return max(0.0, float(residual) - residual_zero_threshold)

    def _score_from_residual(m: FileMetric, residual: float) -> float:
        aocc = max(float(m.structural_aocc), 0.0)
        r_eff = _effective_residual(residual)

        
        return aocc * math.exp(-residual_gain * r_eff)
        

    def _clear_metric(m: FileMetric, mask_source: str, n_tiles_in_mask: int) -> None:
        m.spatial_mask_source = mask_source
        m.n_flicker_tiles = n_tiles_in_mask
        m.residual_ratio = 0.0
        m.residual_linear_ratio = 0.0
        m.effective_residual_ratio = 0.0
        m.band_residual_ratio = 0.0
        m.raw_peak_excess_for_norm = 0.0
        m.db_peak_prominence = 0.0
        m.raw_db_peak_prominence_for_norm = 0.0
        m.score = _score_from_residual(m, 0.0)
        m.peak_excess_energy = 0.0
        m.total_band_energy_in_mask = 0.0
        m.background_median = 0.0
        m.event_ratio_in_mask = 0.0
        m.residual_peak_freqs_hz = []

    def _set_zero(mask_source: str, n_tiles_in_mask: int) -> None:
        for m in metrics:
            _clear_metric(m, mask_source, n_tiles_in_mask)

    if band_freqs is None or band_freqs.size < 3 \
            or raw.band_power_full is None or raw.band_power_full.size == 0:
        _set_zero("no_spectrum", 0)
        return

    n_band = band_freqs.size
    n_in_mask = int(spatial_mask.sum())
    if n_in_mask == 0:
        _set_zero("empty_mask", 0)
        return

    # 1) Detect the dominant flicker peak bins on the RAW aggregate spectrum.
    raw_agg = raw.band_power_full[spatial_mask].sum(axis=0)
    raw_median_for_detection = float(np.median(raw_agg)) + EPS
    raw_excess_for_detection = np.maximum(0.0, raw_agg - raw_median_for_detection)
    threshold_excess = max(0.0, (detection_threshold_factor - 1.0) * raw_median_for_detection)

    local_max = np.zeros(n_band, dtype=bool)
    if n_band >= 3:
        local_max[1:-1] = (raw_agg[1:-1] > raw_agg[:-2]) \
                          & (raw_agg[1:-1] > raw_agg[2:])
    local_max &= (raw_excess_for_detection > threshold_excess)
    candidate_indices = np.where(local_max)[0]

    if candidate_indices.size == 0:
        candidate_indices = np.where(raw_excess_for_detection > 0.0)[0]
        if candidate_indices.size == 0:
            _set_zero(f"global:{n_in_mask}_active_tiles;no_peak_detected", n_in_mask)
            print("  residual: no peak detected on raw aggregate spectrum; residual=0 for all methods.")
            return

    sorted_by_power = candidate_indices[np.argsort(raw_excess_for_detection[candidate_indices])[::-1]]
    min_separation = 2 * peak_half_width_bins + 1
    selected_peaks: List[int] = []
    for idx in sorted_by_power:
        if all(abs(int(idx) - int(s)) > min_separation for s in selected_peaks):
            selected_peaks.append(int(idx))
        if len(selected_peaks) >= n_residual_peaks:
            break

    peak_mask = np.zeros(n_band, dtype=bool)
    for j in selected_peaks:
        lo = max(0, j - peak_half_width_bins)
        hi = min(n_band, j + peak_half_width_bins + 1)
        peak_mask[lo:hi] = True
    background_mask = ~peak_mask
    if not background_mask.any():
        background_mask = np.ones(n_band, dtype=bool)

    selected_freqs_hz = [float(band_freqs[j]) for j in selected_peaks]
    n_peak_bins = int(peak_mask.sum())
    n_bg_bins = int(background_mask.sum())
    print(f"  residual: top-{len(selected_peaks)} peaks at "
          f"{[f'{f:.1f}' for f in selected_freqs_hz]} Hz | "
          f"{n_peak_bins} peak bins (half_width={peak_half_width_bins}) | "
          f"{n_bg_bins}/{n_band} background bins | "
          f"raw-normalized dB peak-over-background residual | "
          f"db_threshold={peak_db_threshold_db:g} dB | "
          f"residual_power={residual_power:g}")

    mask_source = (f"global:{n_in_mask}_active_tiles;"
                   f"top{len(selected_peaks)}_peaks;rawnorm_peakdb_power{residual_power:g}")
    raw_event_sum = float(raw.tile_event_counts[spatial_mask].sum()) + EPS

    # 2) First pass: compute both dB peak-over-background prominence
    #    and linear peak-minus-background excess for diagnostics.
    computed = []
    for m in metrics:
        m.spatial_mask_source = mask_source
        m.n_flicker_tiles = n_in_mask
        m.residual_peak_freqs_hz = selected_freqs_hz

        bp = m.band_power_full
        if bp is None or bp.size == 0:
            computed.append((m, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue

        method_band_freqs = m.band_freqs_full
        if method_band_freqs is None or method_band_freqs.size == 0                 or method_band_freqs.size == n_band:
            method_peak_mask = peak_mask
            method_background_mask = background_mask
        else:
            nn_idx = np.argmin(
                np.abs(band_freqs[np.newaxis, :]
                       - method_band_freqs[:, np.newaxis]),
                axis=1)
            method_peak_mask = peak_mask[nn_idx]
            method_background_mask = background_mask[nn_idx]
            if not method_background_mask.any():
                method_background_mask = np.ones_like(method_peak_mask)

        m_agg = bp[spatial_mask].sum(axis=0)
        if m_agg.size == 0 or float(m_agg.sum()) <= EPS:
            event_ratio = float(m.tile_event_counts[spatial_mask].sum()) / raw_event_sum
            computed.append((m, 0.0, 0.0, 0.0, 0.0, event_ratio))
            continue

        bg_median = float(np.median(m_agg[method_background_mask])) + EPS
        total_band = float(m_agg.sum()) + EPS
        event_ratio = float(m.tile_event_counts[spatial_mask].sum()) / raw_event_sum

        # Diagnostic linear-domain excess.  This is kept for checking absolute
        # FFT power, but it is not used as the headline residual.
        bg_level_for_subtraction = bg_median * (1.0 + peak_excess_bg_threshold_factor)
        linear_peak_excess = float(
            np.maximum(0.0, m_agg[method_peak_mask] - bg_level_for_subtraction).sum()
        )

        # Headline dB-domain prominence.  This matches the spectrum plots:
        # a 20 dB peak means 100x background power, 40 dB means 10000x.
        peak_values = m_agg[method_peak_mask]
        with np.errstate(divide='ignore', invalid='ignore'):
            db_prominence_each_bin = 10.0 * np.log10((peak_values + EPS) / (bg_median + EPS))
        db_excess_each_bin = np.maximum(0.0, db_prominence_each_bin - peak_db_threshold_db)
        db_peak_prominence = float(np.mean(db_excess_each_bin)) if db_excess_each_bin.size else 0.0

        computed.append((m, db_peak_prominence, linear_peak_excess, total_band, bg_median, event_ratio))

    raw_db_peak_prominence = float(computed[0][1])
    raw_linear_peak_excess = float(computed[0][2])
    if raw_db_peak_prominence <= EPS:
        _set_zero(f"{mask_source};raw_db_peak_prominence_zero", n_in_mask)
        print("  residual: raw dB peak prominence is zero; residual=0 for all methods.")
        return

    # 3) Second pass: normalize dB prominence by RAW dB prominence.
    #    linear_residual here means the raw-normalized dB prominence ratio.
    for m, db_peak_prominence, linear_peak_excess, total_band, bg_median, event_ratio in computed:
        linear_residual = float(db_peak_prominence) / (raw_db_peak_prominence + EPS)
        linear_residual = max(0.0, linear_residual)
        residual = linear_residual ** residual_power

        # Make raw exactly 1 for readability and to preserve the original convention.
        if m is raw:
            linear_residual = 1.0
            residual = 1.0

        m.residual_linear_ratio = linear_residual
        
        m.residual_ratio = residual
        if residual > 1:
        

            m.residual_ratio = 1
            residual = 1
        


        m.effective_residual_ratio = _effective_residual(residual)
        m.band_residual_ratio = float(linear_peak_excess) / (float(total_band) + EPS) if total_band > 0 else 0.0
        m.raw_peak_excess_for_norm = raw_linear_peak_excess
        m.db_peak_prominence = float(db_peak_prominence)
        m.raw_db_peak_prominence_for_norm = raw_db_peak_prominence
        m.score = _score_from_residual(m, residual)
        m.peak_excess_energy = float(linear_peak_excess)
        m.total_band_energy_in_mask = float(total_band)
        m.background_median = float(bg_median)
        m.event_ratio_in_mask = float(event_ratio)


# ---------------- CSV writer ---------------- #

def write_combined_metrics_csv(output_csv: Path, metrics: List[FileMetric],
                                 args, score_name: str) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "GroupID", "Method", "Filename", "SourcePath",
        score_name, "StructuralAOCC", "ResidualRatio", "LinearResidualRatio",
        "EffectiveResidualRatio", "BandResidualRatio",
        "DbPeakProminence", "RawDbPeakProminenceForNorm",
        "LinearPeakExcessEnergy", "RawLinearPeakExcessForInfo",
        "TotalBandEnergy", "BackgroundMedian",
        "ResidualPeakFreqsHz", "EventRatioInMask",
        "RelPenaltyTopMean", "DetectedTopPeaksHz",
        "DetectedFreqHz", "SpatialMaskSource", "NumFlickerTiles",
        "Lambda", "ScoreMode", "AOCCAlpha", "ResidualGain", "PenaltyPower",
        "ResidualZeroThreshold", "PeakExcessBgThresholdFactor", "PeakDbThresholdDb",
        "NResidualPeaks", "PeakHalfWidthBins",
        "HighPassCutoffHz", "PeriodicFmaxHz",
        "MicrobinUs", "TileCols", "TileRows", "PeakRatioThreshold",
        "InputFormat", "EventCount",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in metrics:
            peaks_str = ";".join(f"{f:.2f}:{e:.2e}"
                                 for f, e in (m.detected_peaks_hz or [])[:5])
            residual_peaks_str = ";".join(f"{f:.2f}"
                                           for f in (m.residual_peak_freqs_hz or []))
            w.writerow([
                m.group_id, m.method, m.name, m.source_path,
                f"{m.score:.6f}", f"{m.structural_aocc:.6f}", f"{m.residual_ratio:.6f}",
                f"{m.residual_linear_ratio:.6e}",
                f"{m.effective_residual_ratio:.6f}",
                f"{m.band_residual_ratio:.6f}",
                f"{m.db_peak_prominence:.6f}",
                f"{m.raw_db_peak_prominence_for_norm:.6f}",
                f"{m.peak_excess_energy:.6e}",
                f"{m.raw_peak_excess_for_norm:.6e}",
                f"{m.total_band_energy_in_mask:.6e}",
                f"{m.background_median:.6e}",
                residual_peaks_str,
                f"{m.event_ratio_in_mask:.6f}",
                f"{m.rel_penalty:.6f}", peaks_str,
                f"{m.detected_freq_hz:.3f}", m.spatial_mask_source, m.n_flicker_tiles,
                f"{args.lambda_value:.6f}", args.score_mode,
                f"{args.aocc_alpha:.6f}", f"{args.residual_gain:.6f}",
                f"{args.penalty_power:.6f}",
                f"{args.residual_zero_threshold:.6f}",
                f"{args.peak_excess_bg_threshold_factor:.6f}",
                f"{args.peak_db_threshold_db:.6f}",
                args.n_residual_peaks, args.peak_half_width_bins,
                f"{args.hp_cutoff_hz:.3f}", f"{args.periodic_fmax_hz:.3f}",
                f"{args.microbin_us:.3f}",
                args.tile_cols, args.tile_rows,
                f"{args.peak_ratio_threshold:.3f}",
                args.input_format, m.event_count,
            ])
    print(f"\nSummary saved to: {output_csv}")


# ---------------- diagnostic plots ---------------- #

def save_purity_heatmap(values: np.ndarray, tile_rows: int, tile_cols: int,
                          output_path: Path, title: str, vmin=None, vmax=None,
                          mask: np.ndarray = None, cmap: str = "magma") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != tile_rows * tile_cols:
        out = np.zeros(tile_rows * tile_cols, dtype=np.float64)
        out[:min(out.size, arr.size)] = arr[:min(out.size, arr.size)]
        arr = out
    grid = arr.reshape(tile_rows, tile_cols)

    plt.figure(figsize=(max(6, tile_cols * 0.45), max(3, tile_rows * 0.45)))
    plt.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(fraction=0.046, pad=0.04)
    if mask is not None and mask.size == arr.size:
        m_grid = mask.reshape(tile_rows, tile_cols)
        for r in range(tile_rows):
            for c in range(tile_cols):
                if m_grid[r, c]:
                    plt.gca().add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                                       fill=False, edgecolor='cyan',
                                                       linewidth=1.5))
    plt.title(title); plt.xlabel("tile x"); plt.ylabel("tile y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()


# ---------------- global DFT plotting ---------------- #

def compute_global_dft(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                        W: int, H: int, microbin_us: float,
                        fmax_hz: float = 500.0,
                        hp_cutoff_hz: float = 0.0
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if t.size == 0:
        return None

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    t_v = t[valid]
    if t_v.size == 0:
        return None

    duration_us = max(float(t_v[-1] - t_v[0]), float(microbin_us))
    n_bins = max(16, int(np.ceil(duration_us / microbin_us)) + 1)
    bin_idx = np.floor((t_v - t_v[0]) / microbin_us).astype(np.int64)
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)
    if not in_range.any():
        return None
    bin_idx = bin_idx[in_range]

    signal = np.zeros(n_bins, dtype=np.float64)
    np.add.at(signal, bin_idx, 1.0)
    signal -= signal.mean()

    dt_s = microbin_us * 1e-6
    freqs = np.fft.rfftfreq(n_bins, d=dt_s)
    spec = np.fft.rfft(signal)
    power = spec.real ** 2 + spec.imag ** 2
    power_db = 10.0 * np.log10(power + EPS)

    nyquist = 0.5 / dt_s
    band = (freqs >= max(0.0, hp_cutoff_hz)) & (freqs <= min(fmax_hz, nyquist))
    f_band = freqs[band]
    db_band = power_db[band]
    if f_band.size == 0:
        return None
    return f_band, db_band


def find_peak_db_at_frequency(freqs: np.ndarray, power_db: np.ndarray,
                                target_hz: float,
                                search_half_width_hz: float = 4.0
                                ) -> Optional[float]:
    if freqs.size == 0:
        return None
    mask = (freqs >= target_hz - search_half_width_hz) & \
           (freqs <= target_hz + search_half_width_hz)
    if not mask.any():
        return None
    return float(np.max(power_db[mask]))


def compute_shared_dft_ylim(spectra: List[Tuple[np.ndarray, np.ndarray]]
                              ) -> Tuple[float, float]:
    if not spectra:
        return -20.0, 0.0

    medians = [float(np.median(db)) for _, db in spectra]
    maxes = [float(np.max(db)) for _, db in spectra]
    floor = min(medians)
    ceiling = max(maxes)
    span = max(20.0, ceiling - floor)

    y_low = floor - 0.30 * span
    y_high = ceiling + 0.10 * span
    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_high <= y_low:
        y_low, y_high = floor - 10.0, ceiling + 5.0
    return y_low, y_high


def _dft_colour_for_method(method: str) -> str:
    """Each PFD variant gets a deterministic colour shade based on its
    label so multiple PFD lines stay distinguishable."""
    PFD_PALETTE = [
        "#d62728", "#ff7f0e", "#9467bd", "#8c564b",
        "#e377c2", "#bcbd22", "#17becf", "#7f7f7f",
    ]
    m = method.lower()
    if m == "raw":
        return "#1f77b4"
    if m == "efr":
        return "#2ca02c"
    if m == "pfd":
        return PFD_PALETTE[0]
    if m.startswith("pfd:"):
        label = m.split(":", 1)[1]
        idx = (sum(ord(c) for c in label)) % len(PFD_PALETTE)
        return PFD_PALETTE[idx]
    return "#444444"


def render_global_dft_plot(f_plot: np.ndarray, db_plot: np.ndarray,
                            output_path: Path,
                            method: str,
                            y_low: float, y_high: float,
                            fmax_hz: float,
                            ref_db: Optional[float] = None) -> None:
    colour = _dft_colour_for_method(method)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(f_plot, db_plot, linewidth=1.0, color=colour)

    if ref_db is not None and np.isfinite(ref_db):
        ax.axhline(ref_db, color='black', linewidth=1.2, linestyle='--',
                    alpha=0.85)

    ax.set_xlabel("Frequency (Hz)", fontsize=20)
    ax.set_ylabel("Power (dB)", fontsize=20)
    ax.set_xlim(0.0, float(fmax_hz))
    ax.set_ylim(y_low, y_high)
    ax.tick_params(axis='both', which='major', labelsize=16,
                    direction='in', length=4, width=1.0)

    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)
        ax.spines[s].set_color('black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def render_aggregate_spectrum(freqs_hz: np.ndarray, power_lin: np.ndarray,
                                output_path: Path, method: str,
                                y_low: float, y_high: float,
                                fmax_hz: float,
                                ref_db: Optional[float] = None,
                                peak_freqs_hz: Optional[List[float]] = None
                                ) -> None:
    """Aggregate spectrum panel (dB). The top-N peak frequencies used in
    the residual computation are annotated as dotted vertical lines."""
    colour = _dft_colour_for_method(method)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db_plot = 10.0 * np.log10(power_lin + EPS)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(freqs_hz, db_plot, linewidth=1.0, color=colour)

    if ref_db is not None and np.isfinite(ref_db):
        ax.axhline(ref_db, color='black', linewidth=1.2, linestyle='--',
                    alpha=0.85)

    if peak_freqs_hz:
        for pf in peak_freqs_hz:
            ax.axvline(pf, color='gray', linewidth=0.7,
                        linestyle=':', alpha=0.7)

    ax.set_xlabel("Frequency (Hz)", fontsize=20)
    ax.set_ylabel("Aggregate Power (dB)", fontsize=20)
    ax.set_xlim(0.0, float(fmax_hz))
    ax.set_ylim(y_low, y_high)
    ax.tick_params(axis='both', which='major', labelsize=16,
                    direction='in', length=4, width=1.0)

    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)
        ax.spines[s].set_color('black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------- args + main ---------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("AOCC-flicker v3 (peak-vs-background residual): "
                     "1 raw + N PFD variants (+optional EFR). Residual is "
                     "peak excess over same-sequence background, top-N peaks "
                     "only, no normalisation to raw."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # files + IO
    p.add_argument("--raw_file", type=str, required=True)
    p.add_argument("--pfd_files", type=str, nargs="+", required=True,
                   help="One or more PFD-denoised .txt files for the same "
                        "sequence. Each is processed as a separate method "
                        "(pfd:<label>).")
    p.add_argument("--pfd_labels", type=str, nargs="+", default=None,
                   help="Optional short labels matching --pfd_files. "
                        "Default: file stems.")
    p.add_argument("--efr_file", type=str, default="",
                   help="Optional EFR-denoised .txt file (single).")
    p.add_argument("--group_id", type=str, default="")
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--ccc_dir", type=str, required=True)
    p.add_argument("--input_format", type=str, default="auto",
                   choices=["auto", "txyp", "xypt"])
    p.add_argument("--format_tail_lines", type=int, default=2000)
    p.add_argument("--first_col_time_threshold", type=float, default=1100.0)

    # geometry
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--tile_cols", type=int, default=16)
    p.add_argument("--tile_rows", type=int, default=9)

    # AOCC
    p.add_argument("--min_interval_us", type=float, default=4000.0)
    p.add_argument("--max_interval_us", type=float, default=50001.0)
    p.add_argument("--step_us", type=float, default=2000.0)
    p.add_argument("--aocc_x_min", type=float, default=0.0)
    p.add_argument("--aocc_x_max", type=float, default=-1.0)
    p.add_argument("--beta", type=float, default=-1.0)
    p.add_argument("--max_windows_per_interval", type=int, default=0)
    p.add_argument("--contrast_interval_stride", type=int, default=1)
    p.add_argument("--skip_structural_aocc", action="store_true")

    # FFT / per-tile diagnostic pass
    p.add_argument("--microbin_us", type=float, default=500.0)
    p.add_argument("--hp_cutoff_hz", type=float, default=40.0)
    p.add_argument("--periodic_fmax_hz", type=float, default=300.0)
    p.add_argument("--peak_ratio_threshold", type=float, default=8.0)
    p.add_argument("--diagnostic_top_k", type=int, default=5)
    p.add_argument("--min_events_per_tile", type=int, default=100)
    p.add_argument("--peak_dilation_bins", type=int, default=2)

    # Headline residual params (peak-vs-background)
    p.add_argument("--n_residual_peaks", type=int, default=3,
                   help="Number of top peaks (detected on raw aggregate "
                        "spectrum) used in the headline residual. Default "
                        "3 — keep small so only the strongest tones count.")
    p.add_argument("--peak_half_width_bins", type=int, default=2,
                   help="Half-width (FFT bins) around each top-N peak "
                        "treated as part of the peak region.")
    p.add_argument("--peak_detection_threshold_factor", type=float, default=2.0,
                   help="Peak candidate iff raw_agg(f) > factor * median(raw_agg).")
    p.add_argument("--peak_excess_bg_threshold_factor", type=float, default=0.0,
                   help=("Diagnostic only in this dB version: extra background margin for linear "
                         "peak-minus-background excess. It does not affect the headline dB residual."))
    p.add_argument("--peak_db_threshold_db", type=float, default=0.0,
                   help=("dB margin subtracted from peak-over-background prominence. "
                         "For example, 3 means only peak bins more than 3 dB above background contribute."))
    p.add_argument("--residual_zero_threshold", type=float, default=0.0,
                   help=("Residual dead-zone used only for score. ResidualRatio itself stays raw-normalized, "
                         "so raw remains 1.0 when raw dB peak prominence exists."))
    p.add_argument("--residual_power", type=float, default=1.0,
                   help=("Optional power compression for raw-normalized dB residual. "
                         "Use 1.0 by default because the dB transform already compresses dynamic range."))

    # Score
    p.add_argument("--score_mode", type=str, default="raw_aocc_exp",
                   choices=["raw_aocc_exp", "flicker_first", "legacy_divide", "power_divide", "exp"],
                   help=("Scoring mode. raw_aocc_exp uses raw AOCC as sequence scale and ranks by "
                         "raw-normalized flicker residual. legacy_divide keeps the older AOCC/(1+lambda*R) form."))
    p.add_argument("--lambda_value", type=float, default=0.5,
                   help="Penalty strength for legacy_divide/power_divide modes.")
    p.add_argument("--aocc_alpha", type=float, default=0.5,
                   help="AOCC compression exponent for flicker_first. Smaller => AOCC gaps matter less.")
    p.add_argument("--residual_gain", type=float, default=3.0,
                   help="Exponential residual penalty for flicker_first/exp. Larger => residual flicker matters more.")
    p.add_argument("--penalty_power", type=float, default=1.0,
                   help="Power for power_divide score mode.")

    # misc
    p.add_argument("--max_events", type=int, default=0)
    p.add_argument("--diagnostic_dir", type=str, default="")
    p.add_argument("--no_diagnostic_maps", action="store_true")
    p.add_argument("--global_dft_plots", action="store_true")
    p.add_argument("--dft_dir", type=str, default="")
    p.add_argument("--dft_fmax_hz", type=float, default=500.0)
    p.add_argument("--dft_ref_freq_hz", type=float, default=100.0)
    p.add_argument("--dft_ref_half_width_hz", type=float, default=4.0)

    return p.parse_args()


def derive_pfd_labels_for_files(pfd_files: List[Path],
                                   provided_labels: Optional[List[str]] = None
                                   ) -> List[str]:
    if provided_labels:
        if len(provided_labels) != len(pfd_files):
            raise ValueError(
                f"--pfd_labels has {len(provided_labels)} entries but "
                f"--pfd_files has {len(pfd_files)} entries.")
        labels = list(provided_labels)
    else:
        labels = [p.stem for p in pfd_files]

    seen: dict = {}
    final: List[str] = []
    for lbl in labels:
        if lbl in seen:
            seen[lbl] += 1
            final.append(f"{lbl}_{seen[lbl]}")
        else:
            seen[lbl] = 1
            final.append(lbl)
    return final


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_file)
    pfd_paths = [Path(p) for p in args.pfd_files]
    pfd_labels = derive_pfd_labels_for_files(pfd_paths, args.pfd_labels)
    efr_path = Path(args.efr_file) if args.efr_file else None

    for p in [raw_path] + pfd_paths + ([efr_path] if efr_path else []):
        if not p.exists():
            raise FileNotFoundError(f"input file does not exist: {p}")

    group_id = args.group_id if args.group_id else raw_path.stem

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    print(f"group_id     : {group_id}")
    print(f"raw          : {raw_path}")
    for lbl, fp in zip(pfd_labels, pfd_paths):
        print(f"pfd[{lbl}]    : {fp}")
    if efr_path is not None:
        print(f"efr          : {efr_path}")
    if args.score_mode == "legacy_divide":
        print(f"score formula: AOCC / (1 + lambda * effective_residual), lambda={args.lambda_value}")
    elif args.score_mode == "power_divide":
        print(f"score formula: AOCC / (1 + lambda * effective_residual)^power, "
              f"lambda={args.lambda_value}, power={args.penalty_power}")
    elif args.score_mode == "exp":
        print(f"score formula: AOCC * exp(-gain * effective_residual), gain={args.residual_gain}")
    elif args.score_mode == "flicker_first":
        print(f"score formula: raw_AOCC * (AOCC/raw_AOCC)^alpha * "
              f"exp(-gain * effective_residual), alpha={args.aocc_alpha}, "
              f"gain={args.residual_gain}")
    else:
        print(f"score formula: raw_AOCC * exp(-gain * effective_residual), "
              f"gain={args.residual_gain}")
    print(f"residual     : raw-normalized peak-minus-background excess; "
          f"dB ResidualRatio(raw)=1, top-{args.n_residual_peaks} peaks "
          f"(half_width={args.peak_half_width_bins}), "
          f"bg_margin={args.peak_excess_bg_threshold_factor}, "
          f"zero_threshold={args.residual_zero_threshold}")
    print(f"mode         : GLOBAL (all tiles, no ROI)")
    print(f"pandas available: {HAVE_PANDAS}")

    diagnostic_dir = Path(args.diagnostic_dir) if args.diagnostic_dir \
                      else Path(args.ccc_dir) / "diagnostic_maps"
    dft_dir = Path(args.dft_dir) if args.dft_dir \
              else Path(args.ccc_dir) / "global_dft"
    if args.global_dft_plots:
        print(f"global DFT plots: enabled -> {dft_dir}")

    t_total = time.time()

    # Process raw first; lock in its per-tile peak mask for downstream
    # per-tile DIAGNOSTIC computations. (The headline residual uses its
    # own top-N peak detection on the aggregate spectrum, independent of
    # this per-tile mask.)
    ref = process_single_path(args, raw_path, group_id, "raw",
                                compute_dft=args.global_dft_plots,
                                peak_mask_override=None)
    raw_peak_override = None
    if ref.peak_mask_full is not None and ref.band_freqs_full is not None \
            and ref.band_freqs_full.size > 0:
        raw_peak_override = (ref.band_freqs_full, ref.peak_mask_full)

    pfd_metrics: List[FileMetric] = []
    for label, pp in zip(pfd_labels, pfd_paths):
        method_tag = f"pfd:{label}"
        pfd_m = process_single_path(args, pp, group_id, method_tag,
                                      compute_dft=args.global_dft_plots,
                                      peak_mask_override=raw_peak_override)
        pfd_metrics.append(pfd_m)

    efr_m: Optional[FileMetric] = None
    if efr_path is not None:
        efr_m = process_single_path(args, efr_path, group_id, "efr",
                                      compute_dft=args.global_dft_plots,
                                      peak_mask_override=raw_peak_override)

    gm: List[FileMetric] = [ref] + pfd_metrics
    if efr_m is not None:
        gm.append(efr_m)

    n_tiles = args.tile_cols * args.tile_rows
    spatial_mask = ref.tile_event_counts > 0
    print(f"  spatial mask: GLOBAL ({int(spatial_mask.sum())}/{n_tiles} active tiles)")

    compute_peak_vs_background_residual(
        metrics=gm,
        spatial_mask=spatial_mask,
        lambda_value=args.lambda_value,
        n_residual_peaks=int(args.n_residual_peaks),
        peak_half_width_bins=int(args.peak_half_width_bins),
        detection_threshold_factor=float(args.peak_detection_threshold_factor),
        score_mode=args.score_mode,
        aocc_alpha=float(args.aocc_alpha),
        residual_gain=float(args.residual_gain),
        penalty_power=float(args.penalty_power),
        residual_zero_threshold=float(args.residual_zero_threshold),
        peak_excess_bg_threshold_factor=float(args.peak_excess_bg_threshold_factor),
        peak_db_threshold_db=float(args.peak_db_threshold_db),
        residual_power=float(args.residual_power),
    )

    for m in gm:
        print(f"  {m.method:>16s}/{m.name}: "
              f"AOCC={m.structural_aocc:.4f}, "
              f"Residual={m.residual_ratio:.4f}, "
              f"LinearR={m.residual_linear_ratio:.3e}, "
              f"EffResidual={m.effective_residual_ratio:.4f}, "
              f"BandR={m.band_residual_ratio:.4f}, "
              f"DbPeak={m.db_peak_prominence:.2f}dB, "
              f"LinPeakExcess={m.peak_excess_energy:.2e}, "
              f"TotalBand={m.total_band_energy_in_mask:.2e}, "
              f"BgMedian={m.background_median:.2e}, "
              f"EventRatio={m.event_ratio_in_mask:.4f}, "
              f"score={m.score:.4f}")

    if not args.no_diagnostic_maps:
        for m in gm:
            title = (f"group {group_id} {m.method} flicker_purity "
                     f"(rel_pen={m.rel_penalty:.3f})")
            safe_method = m.method.replace(":", "_")
            save_purity_heatmap(
                m.flicker_purity_per_tile,
                args.tile_rows, args.tile_cols,
                diagnostic_dir / f"group_{group_id}" /
                f"{safe_method}_flicker_purity.png",
                title, vmin=0.0, vmax=1.0, mask=spatial_mask,
            )

    # global DFT panels
    if args.global_dft_plots:
        raw_ref_db = None
        if ref.dft_freqs_hz is not None and ref.dft_power_db is not None:
            raw_ref_db = find_peak_db_at_frequency(
                ref.dft_freqs_hz, ref.dft_power_db,
                target_hz=float(args.dft_ref_freq_hz),
                search_half_width_hz=float(args.dft_ref_half_width_hz),
            )

        raw_plus_pfd = [ref] + pfd_metrics
        shared_spectra = [(m.dft_freqs_hz, m.dft_power_db)
                           for m in raw_plus_pfd
                           if m.dft_freqs_hz is not None
                           and m.dft_power_db is not None]
        if shared_spectra:
            shared_low, shared_high = compute_shared_dft_ylim(shared_spectra)
            for m in raw_plus_pfd:
                if m.dft_freqs_hz is None or m.dft_power_db is None:
                    continue
                safe_method = m.method.replace(":", "_")
                render_global_dft_plot(
                    f_plot=m.dft_freqs_hz, db_plot=m.dft_power_db,
                    output_path=(dft_dir / f"group_{group_id}" /
                                 f"{safe_method}_global_dft.png"),
                    method=m.method,
                    y_low=shared_low, y_high=shared_high,
                    fmax_hz=float(args.dft_fmax_hz),
                    ref_db=raw_ref_db,
                )

        if efr_m is not None and efr_m.dft_freqs_hz is not None \
                and efr_m.dft_power_db is not None:
            efr_low, efr_high = compute_shared_dft_ylim(
                [(efr_m.dft_freqs_hz, efr_m.dft_power_db)])
            render_global_dft_plot(
                f_plot=efr_m.dft_freqs_hz, db_plot=efr_m.dft_power_db,
                output_path=(dft_dir / f"group_{group_id}" /
                             f"{efr_m.method}_global_dft.png"),
                method=efr_m.method,
                y_low=efr_low, y_high=efr_high,
                fmax_hz=float(args.dft_fmax_hz),
                ref_db=raw_ref_db,
            )

    # aggregate spectrum panels — top-N peaks marked with dotted lines
    agg_dir = Path(args.ccc_dir) / "aggregate_spectrum"
    raw_plus_pfd = [ref] + pfd_metrics
    agg_spectra_for_ylim = []
    for m in raw_plus_pfd:
        if m.aggregate_power is not None and m.aggregate_power.size > 0:
            db = 10.0 * np.log10(m.aggregate_power + EPS)
            agg_spectra_for_ylim.append((m.aggregate_freqs_hz, db))

    if agg_spectra_for_ylim:
        agg_low, agg_high = compute_shared_dft_ylim(agg_spectra_for_ylim)
        agg_ref_db = None
        if ref.aggregate_power is not None and ref.aggregate_power.size > 0:
            ref_db_array = 10.0 * np.log10(ref.aggregate_power + EPS)
            agg_ref_db = find_peak_db_at_frequency(
                ref.aggregate_freqs_hz, ref_db_array,
                target_hz=float(args.dft_ref_freq_hz),
                search_half_width_hz=float(args.dft_ref_half_width_hz),
            )

        residual_peak_freqs = ref.residual_peak_freqs_hz or []

        for m in raw_plus_pfd:
            if m.aggregate_power is None or m.aggregate_power.size == 0:
                continue
            safe_method = m.method.replace(":", "_")
            render_aggregate_spectrum(
                freqs_hz=m.aggregate_freqs_hz,
                power_lin=m.aggregate_power,
                output_path=(agg_dir / f"group_{group_id}" /
                             f"{safe_method}_aggregate_spectrum.png"),
                method=m.method,
                y_low=agg_low, y_high=agg_high,
                fmax_hz=float(args.periodic_fmax_hz),
                ref_db=agg_ref_db,
                peak_freqs_hz=residual_peak_freqs,
            )

        if efr_m is not None and efr_m.aggregate_power is not None \
                and efr_m.aggregate_power.size > 0:
            efr_db_array = 10.0 * np.log10(efr_m.aggregate_power + EPS)
            efr_agg_low, efr_agg_high = compute_shared_dft_ylim(
                [(efr_m.aggregate_freqs_hz, efr_db_array)])
            render_aggregate_spectrum(
                freqs_hz=efr_m.aggregate_freqs_hz,
                power_lin=efr_m.aggregate_power,
                output_path=(agg_dir / f"group_{group_id}" /
                             f"{efr_m.method}_aggregate_spectrum.png"),
                method=efr_m.method,
                y_low=efr_agg_low, y_high=efr_agg_high,
                fmax_hz=float(args.periodic_fmax_hz),
                ref_db=agg_ref_db,
                peak_freqs_hz=residual_peak_freqs,
            )
        print(f"  aggregate spectrum plots written to {agg_dir}/group_{group_id}/")

    # release transient buffers
    for m in gm:
        m.dft_freqs_hz = None
        m.dft_power_db = None
        m.aggregate_freqs_hz = None
        m.aggregate_power = None
        m.band_power_full = None
        m.band_freqs_full = None
        m.active_tile_mask = None
        m.peak_mask_full = None

    write_combined_metrics_csv(
        Path(args.output_csv), gm, args,
        score_name=f"AOCC-flicker-rawnorm-peakdiff-{args.score_mode}-global",
    )
    print(f"\nTotal: {time.time() - t_total:.1f}s for group {group_id}.")


if __name__ == "__main__":
    main()
