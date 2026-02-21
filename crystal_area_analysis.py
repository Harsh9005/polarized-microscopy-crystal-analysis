#!/usr/bin/env python3
"""
Crystal Area Quantification from Polarized Light Microscopy Images
===================================================================

Quantifies the area fraction covered by birefringent crystals in polarized
light microscopy images of crystalline samples over time.

Method
------
Under crossed polarizers, amorphous (isotropic) regions appear dark while
crystalline domains exhibit birefringence and transmit light as bright
regions.  This script segments bright (crystalline) pixels from the dark
background using an adaptive threshold on a selected colour channel and
reports the crystal-covered area as a percentage of the total field of view.

Thresholding strategy
~~~~~~~~~~~~~~~~~~~~~
The threshold is computed adaptively for each image as::

    T = max(bg_median + k * MAD, T_min)

where ``bg_median`` and ``MAD`` are the median and median absolute deviation
of the dark pixel sub-population (intensity < 128), ``k`` is a sensitivity
multiplier (default 2), and ``T_min`` is a noise floor (default 10/255).

Image requirements
~~~~~~~~~~~~~~~~~~
- Polarized light microscopy images in ``.tif`` format (RGBA or RGB).
- Filename convention: ``<label> polarized.tif``
  (e.g. ``sample_1 polarized.tif``, ``0 hour polarized.tif``).

Outputs
~~~~~~~
- CSV file with quantitative results per image.
- Composite figure showing original images with segmentation overlay.
- Individual binary mask images for each time point.
- Summary bar chart of crystal area fraction vs. sample.
- Analysis parameters log.

Author:  Harshvardhan Modh
License: MIT
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology, measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def parse_timepoint(filename: str) -> tuple:
    """
    Extract a numeric sort key and a display label from *filename*.

    Recognised patterns::

        "0 hour polarized.tif"   -> (0.0,   "0 h")
        "24 hour polarized.tif"  -> (24.0,  "24 h")
        "Day 5 polarized.tif"    -> (120.0, "Day 5 (120 h)")
        "sample_1 polarized.tif" -> (1.0,   "Sample 1")
        "sample_1.tif"           -> (1.0,   "Sample 1")

    Parameters
    ----------
    filename : str
        Image filename (basename, not full path).

    Returns
    -------
    sort_key : float
        Numeric value used for ordering.
    label : str
        Human-readable label for plots and tables.
    """
    base = os.path.splitext(filename)[0].lower()

    # "X hour" pattern
    match_hour = re.search(r"(\d+)\s*hour", base)
    if match_hour:
        hours = float(match_hour.group(1))
        return hours, f"{int(hours)} h"

    # "Day X" pattern
    match_day = re.search(r"day\s*(\d+)", base)
    if match_day:
        days = float(match_day.group(1))
        hours = days * 24.0
        return hours, f"Day {int(days)} ({int(hours)} h)"

    # "sample_X" pattern
    match_sample = re.search(r"sample[_\s]*(\d+)", base)
    if match_sample:
        idx = float(match_sample.group(1))
        return idx, f"Sample {int(idx)}"

    # Fallback: use filename as label, sort alphabetically
    return 0.0, base


def load_image(filepath: str) -> np.ndarray:
    """Load a ``.tif`` image and return as an RGB ``uint8`` numpy array."""
    img = Image.open(filepath)
    return np.array(img.convert("RGB"), dtype=np.uint8)


def create_scale_bar_mask(shape: tuple, margin_frac: float) -> np.ndarray:
    """
    Boolean mask excluding the scale-bar region (bottom-right corner).

    Parameters
    ----------
    shape : tuple
        ``(H, W)`` or ``(H, W, C)``.
    margin_frac : float
        Fraction of height/width to exclude.

    Returns
    -------
    mask : np.ndarray
        ``True`` = valid pixel, ``False`` = excluded.
    """
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=bool)
    y_start = int(h * (1.0 - margin_frac))
    x_start = int(w * (1.0 - margin_frac * 2))
    mask[y_start:, x_start:] = False
    return mask


def extract_signal_channel(img: np.ndarray, channel: str) -> np.ndarray:
    """
    Extract the analysis channel from an RGB image.

    Parameters
    ----------
    img : np.ndarray
        RGB image ``(H, W, 3)``, ``uint8``.
    channel : str
        ``"green"``, ``"brightness"``, or ``"grayscale"``.

    Returns
    -------
    signal : np.ndarray
        Single-channel ``float64`` image in 0--255 scale.
    """
    if channel == "green":
        return img[:, :, 1].astype(np.float64)
    if channel == "brightness":
        return np.max(img.astype(np.float64), axis=2)
    if channel == "grayscale":
        return (
            0.2989 * img[:, :, 0].astype(np.float64)
            + 0.5870 * img[:, :, 1].astype(np.float64)
            + 0.1140 * img[:, :, 2].astype(np.float64)
        )
    raise ValueError(f"Unknown channel: {channel}")


def compute_background_stats(signal: np.ndarray,
                             valid_mask: np.ndarray) -> dict:
    """
    Background statistics from the dark (non-crystal) pixel population.

    Pixels with intensity < 128 in the valid region are treated as
    background.  This ensures a consistent estimate regardless of how much
    of the field is occupied by bright crystalline domains.

    Returns
    -------
    dict
        ``bg_median``, ``bg_mad``, ``dark_fraction``.
    """
    valid_pixels = signal[valid_mask]
    dark_pixels = valid_pixels[valid_pixels < 128]

    if len(dark_pixels) > 100:
        bg_median = np.median(dark_pixels)
        bg_mad = np.median(np.abs(dark_pixels - bg_median))
    else:
        bg_median = 0.0
        bg_mad = 0.0

    dark_fraction = len(dark_pixels) / len(valid_pixels)
    return {
        "bg_median": bg_median,
        "bg_mad": bg_mad,
        "dark_fraction": dark_fraction,
    }


def compute_adaptive_threshold(bg_stats: dict,
                               k: float = 2,
                               min_threshold: float = 10) -> tuple:
    """
    Compute the segmentation threshold adaptively::

        T = max(bg_median + k * max(MAD, 1), min_threshold)

    Returns
    -------
    threshold : float
        Value in 0--255 scale.
    method_desc : str
        Human-readable description.
    """
    bg_median = bg_stats["bg_median"]
    bg_mad = bg_stats["bg_mad"]
    threshold = bg_median + k * max(bg_mad, 1.0)
    threshold = max(threshold, min_threshold)
    method_desc = (
        f"Adaptive (bg_median={bg_median:.0f} + "
        f"{k}\u00d7MAD={bg_mad:.0f}, floor={min_threshold})"
    )
    return threshold, method_desc


def segment_crystals(signal: np.ndarray,
                     valid_mask: np.ndarray,
                     threshold: float,
                     min_size: int,
                     morph_radius: int) -> np.ndarray:
    """
    Segment crystalline regions by thresholding + morphological cleanup.

    Steps
    -----
    1. Threshold signal > T.
    2. Restrict to valid region.
    3. Morphological opening (removes salt noise).
    4. Remove small connected components.

    Returns
    -------
    crystal_mask : np.ndarray
        Boolean mask (``True`` = crystal pixel).
    """
    binary = (signal > threshold) & valid_mask

    if morph_radius > 0:
        binary = morphology.opening(binary, morphology.disk(morph_radius))

    if min_size > 0:
        binary = morphology.remove_small_objects(binary, min_size=min_size)

    return binary


def quantify_crystal_area(crystal_mask: np.ndarray,
                          valid_mask: np.ndarray) -> dict:
    """Compute area statistics from the segmented crystal mask."""
    total = valid_mask.sum()
    crystals = crystal_mask.sum()
    frac = crystals / total if total > 0 else 0.0

    labeled = measure.label(crystal_mask)
    n_objects = labeled.max()
    areas = [r.area for r in measure.regionprops(labeled)] if n_objects else []

    return {
        "total_valid_pixels": int(total),
        "crystal_pixels": int(crystals),
        "area_fraction": frac,
        "area_percent": frac * 100.0,
        "n_crystal_objects": n_objects,
        "mean_object_area_px": float(np.mean(areas)) if areas else 0.0,
        "median_object_area_px": float(np.median(areas)) if areas else 0.0,
        "max_object_area_px": float(np.max(areas)) if areas else 0.0,
    }


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================


def run_analysis(input_dir: str,
                 output_dir: str | None = None,
                 channel: str = "green",
                 k: float = 2,
                 min_threshold: float = 10,
                 scale_bar_margin: float = 0.08,
                 min_object_size: int = 5,
                 morph_radius: int = 1,
                 dpi: int = 300,
                 fmt: str = "tiff") -> pd.DataFrame:
    """
    Run the full crystal-area quantification pipeline.

    Parameters
    ----------
    input_dir : str
        Directory containing polarized ``.tif`` images.
    output_dir : str or None
        Output directory (created if absent).  Defaults to
        ``<input_dir>/analysis_output``.
    channel : str
        Signal channel (``"green"``, ``"brightness"``, ``"grayscale"``).
    k : float
        Sensitivity multiplier for adaptive threshold.
    min_threshold : float
        Noise-floor threshold (0--255).
    scale_bar_margin : float
        Fraction of image to exclude at bottom-right for scale bar.
    min_object_size : int
        Minimum crystal object size (pixels).
    morph_radius : int
        Disk radius for morphological opening.
    dpi : int
        Figure resolution.
    fmt : str
        Output figure format (``"tiff"``, ``"png"``, ``"pdf"``).

    Returns
    -------
    pd.DataFrame
        Table of quantification results.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path / "analysis_output"
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Discover polarized images
    # ------------------------------------------------------------------
    all_tifs = [
        f for f in os.listdir(input_path)
        if f.lower().endswith(".tif")
        and not f.startswith(".")
    ]
    # Prefer files with "polarized" in the name; fall back to all .tif
    polarized_files = [f for f in all_tifs if "polarized" in f.lower()]
    if not polarized_files:
        polarized_files = all_tifs

    polarized_files.sort(key=lambda f: parse_timepoint(f)[0])

    if not polarized_files:
        raise FileNotFoundError(
            f"No .tif files found in {input_dir}"
        )

    print(f"Found {len(polarized_files)} image(s):")
    for f in polarized_files:
        _, label = parse_timepoint(f)
        print(f"  {f}  \u2192  {label}")
    print()

    # ------------------------------------------------------------------
    # 2. Process each image
    # ------------------------------------------------------------------
    results = []

    for fname in polarized_files:
        sort_key, label = parse_timepoint(fname)
        filepath = str(input_path / fname)
        print(f"Processing: {fname} ({label})")

        img = load_image(filepath)
        h, w = img.shape[:2]
        valid_mask = create_scale_bar_mask(img.shape, scale_bar_margin)
        signal = extract_signal_channel(img, channel)

        bg_stats = compute_background_stats(signal, valid_mask)
        print(
            f"  Background: median={bg_stats['bg_median']:.0f}, "
            f"MAD={bg_stats['bg_mad']:.0f}, "
            f"dark frac={bg_stats['dark_fraction']:.1%}"
        )

        threshold, method_desc = compute_adaptive_threshold(
            bg_stats, k=k, min_threshold=min_threshold
        )
        print(f"  Threshold: {threshold:.1f}/255  ({method_desc})")

        crystal_mask = segment_crystals(
            signal, valid_mask, threshold, min_object_size, morph_radius
        )

        stats = quantify_crystal_area(crystal_mask, valid_mask)
        print(f"  Crystal area: {stats['area_percent']:.4f}%")
        print(f"  Crystal objects: {stats['n_crystal_objects']}\n")

        # Save binary mask
        mask_fname = Path(fname).stem + "_mask.tif"
        Image.fromarray(
            (crystal_mask * 255).astype(np.uint8)
        ).save(str(output_path / mask_fname))

        results.append({
            "filename": fname,
            "sort_key": sort_key,
            "label": label,
            "image_width_px": w,
            "image_height_px": h,
            "threshold_method": method_desc,
            "threshold_value": threshold,
            "bg_median": bg_stats["bg_median"],
            "bg_mad": bg_stats["bg_mad"],
            "dark_fraction": bg_stats["dark_fraction"],
            "total_valid_pixels": stats["total_valid_pixels"],
            "crystal_pixels": stats["crystal_pixels"],
            "crystal_area_percent": stats["area_percent"],
            "n_crystal_objects": stats["n_crystal_objects"],
            "mean_object_area_px": stats["mean_object_area_px"],
            "median_object_area_px": stats["median_object_area_px"],
            "max_object_area_px": stats["max_object_area_px"],
        })

    df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # 3. Save CSV
    # ------------------------------------------------------------------
    csv_path = output_path / "crystal_area_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Results saved to: {csv_path}")

    # ------------------------------------------------------------------
    # 4. Composite figure (original + overlay)
    # ------------------------------------------------------------------
    n = len(df)
    if n > 0:
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), dpi=dpi)
        if n == 1:
            axes = axes.reshape(2, 1)

        for idx, (_, row) in enumerate(df.iterrows()):
            img = load_image(str(input_path / row["filename"]))
            valid_mask = create_scale_bar_mask(img.shape, scale_bar_margin)
            signal = extract_signal_channel(img, channel)
            bg = compute_background_stats(signal, valid_mask)
            thr, _ = compute_adaptive_threshold(bg, k=k, min_threshold=min_threshold)
            cmask = segment_crystals(
                signal, valid_mask, thr, min_object_size, morph_radius
            )

            axes[0, idx].imshow(img)
            axes[0, idx].set_title(row["label"], fontsize=11, fontweight="bold")
            axes[0, idx].axis("off")

            overlay = np.clip(img.astype(np.float64) / 255.0 * 2.5, 0, 1)
            overlay[cmask, 0] = 1.0
            overlay[cmask, 1] = 0.0
            overlay[cmask, 2] = 0.0

            axes[1, idx].imshow(overlay)
            axes[1, idx].set_title(
                f"Crystals: {row['crystal_area_percent']:.2f}%", fontsize=10
            )
            axes[1, idx].axis("off")

        fig.suptitle(
            "Crystal Area Quantification \u2014 Polarized Microscopy",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        fig.savefig(
            output_path / f"composite_segmentation.{fmt}",
            dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.1,
        )
        plt.close(fig)
        print(f"Composite figure saved.")

    # ------------------------------------------------------------------
    # 5. Bar chart
    # ------------------------------------------------------------------
    if n > 0:
        fig2, ax = plt.subplots(figsize=(max(5, 1.5 * n), 4.5), dpi=dpi)
        y = df["crystal_area_percent"].values
        labels = df["label"].values

        bars = ax.bar(
            range(n), y, color="#2E86AB",
            edgecolor="black", linewidth=0.8, width=0.6,
        )
        for b, v in zip(bars, y):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + max(y) * 0.02,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=8,
            )

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlabel("Sample", fontsize=12)
        ax.set_ylabel("Crystal Area (%)", fontsize=12)
        ax.set_title("Crystal Coverage", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fig2.savefig(
            output_path / f"crystal_area_chart.{fmt}",
            dpi=dpi, bbox_inches="tight", facecolor="white",
        )
        plt.close(fig2)
        print(f"Bar chart saved.")

    # ------------------------------------------------------------------
    # 6. Parameters log
    # ------------------------------------------------------------------
    log_path = output_path / "analysis_parameters.txt"
    with open(log_path, "w") as fh:
        fh.write("Crystal Area Quantification \u2014 Analysis Parameters\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Input directory:           {input_dir}\n")
        fh.write(f"Output directory:          {output_path}\n\n")
        fh.write("Segmentation settings:\n")
        fh.write(f"  Signal channel:          {channel}\n")
        fh.write(f"  Threshold method:        Adaptive (median + k\u00d7MAD)\n")
        fh.write(f"    k (sensitivity):       {k}\n")
        fh.write(f"    Minimum threshold:     {min_threshold}/255\n")
        fh.write(f"  Scale bar margin (frac): {scale_bar_margin}\n")
        fh.write(f"  Min object size (px):    {min_object_size}\n")
        fh.write(f"  Morph. opening radius:   {morph_radius}\n\n")
        fh.write("Per-image results:\n")
        fh.write("-" * 60 + "\n")
        for _, row in df.iterrows():
            fh.write(f"  {row['filename']}\n")
            fh.write(f"    BG median:  {row['bg_median']:.0f},"
                     f"  MAD: {row['bg_mad']:.0f}\n")
            fh.write(f"    Threshold:  {row['threshold_value']:.1f}/255"
                     f"  ({row['threshold_method']})\n")
            fh.write(f"    Crystal %:  {row['crystal_area_percent']:.4f}\n")
            fh.write(f"    N objects:  {row['n_crystal_objects']}\n\n")

    print(f"Parameters log saved to: {log_path}\nDone.")
    return df


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Quantify birefringent crystal area from polarized light "
            "microscopy images using adaptive thresholding."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --input images/\n"
            "  %(prog)s --input images/ --output results/ --k 3 --channel brightness\n"
            "  %(prog)s --input images/ --format png --dpi 150\n"
        ),
    )
    p.add_argument(
        "-i", "--input", required=True,
        help="Directory containing polarized .tif images.",
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: <input>/analysis_output).",
    )
    p.add_argument(
        "--channel", default="green",
        choices=["green", "brightness", "grayscale"],
        help="Signal channel to segment (default: green).",
    )
    p.add_argument(
        "--k", type=float, default=2,
        help="MAD multiplier for adaptive threshold (default: 2).",
    )
    p.add_argument(
        "--min-threshold", type=float, default=10,
        help="Noise-floor threshold in 0-255 scale (default: 10).",
    )
    p.add_argument(
        "--scale-bar-margin", type=float, default=0.08,
        help="Fraction of image to mask at bottom-right (default: 0.08).",
    )
    p.add_argument(
        "--min-size", type=int, default=5,
        help="Minimum crystal object size in pixels (default: 5).",
    )
    p.add_argument(
        "--morph-radius", type=int, default=1,
        help="Morphological opening disk radius (default: 1).",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Figure resolution (default: 300).",
    )
    p.add_argument(
        "--format", dest="fmt", default="tiff",
        choices=["tiff", "png", "pdf"],
        help="Output figure format (default: tiff).",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    results_df = run_analysis(
        input_dir=args.input,
        output_dir=args.output,
        channel=args.channel,
        k=args.k,
        min_threshold=args.min_threshold,
        scale_bar_margin=args.scale_bar_margin,
        min_object_size=args.min_size,
        morph_radius=args.morph_radius,
        dpi=args.dpi,
        fmt=args.fmt,
    )
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        results_df[["label", "crystal_area_percent",
                     "n_crystal_objects", "threshold_value"]]
        .to_string(index=False)
    )
