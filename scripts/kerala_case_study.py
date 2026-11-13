"""Milestone 5: real-world case study - August 2018 Kerala floods (Kuttanad / Alappuzha).

Pipeline
1. Query Microsoft Planetary Computer STAC for Sentinel-1 RTC (descending orbit 165):
   flood scene 2018-08-21 (two frames, mosaicked) and a dry-season reference
   (per-pixel median of 2018-02-22, 03-06, 03-18 to suppress speckle).
2. Reproject onto Sen1Floods11's grid (EPSG:4326, ~10 m), convert gamma0 to dB.
3. Predict with the 3-seed SAR U-Net ensemble (official regime) via the same tiled
   inference as ``floodseg predict``; write GeoTIFFs.
4. Flood extent = water on 2018-08-21 that is not water in the dry-season reference.
5. Independent cross-check on 2018-08-22 Sentinel-2 L2A (31% cloud): agreement with
   MNDWI and with the Scene Classification water class, on cloud-free pixels only.
6. Folium web map + static figure + JSON/markdown summary.

Caveats are written into the report: Planetary Computer provides terrain-corrected
gamma0 (RTC) whereas Sen1Floods11 uses GEE sigma0 GRD; the S1 pass is ~4 days after
the flood peak; there is no hand-labelled ground truth for this scene.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from floodseg.inference import load_model_from_checkpoint, predict_tiled
from floodseg.predict import write_geotiff

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Kerala")

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX = (76.28, 9.28, 76.58, 9.62)  # lon_min, lat_min, lon_max, lat_max
RES = 8.983152841195215e-05  # Sen1Floods11 pixel size (degrees, ~10 m)
FLOOD_DATE = "2018-08-21"
REF_DATES = ["2018-02-22", "2018-03-06", "2018-03-18"]
ORBIT = 165
S2_CHECK_DATE = "2018-08-22"
S2_PEAK_DATE = "2018-08-17"
MNDWI_TAU = 0.15  # tuned on Sen1Floods11 official train (L1C); applied here to L2A
OUT = Path("outputs/kerala")
RESULTS = Path("outputs/results")
FIGS = Path("outputs/figures")
CKPTS = [Path(f"outputs/runs/official_s1_bce_dice_seed{s}/best_model.pt") for s in (42, 1337, 2024)]

# Reference palette: categorical slots 1 (blue) and 2 (orange).
BLUE, ORANGE, INK2, SURFACE = "#2a78d6", "#eb6834", "#52514e", "#fcfcfb"


def target_grid():
    w = int(round((BBOX[2] - BBOX[0]) / RES))
    h = int(round((BBOX[3] - BBOX[1]) / RES))
    return {"crs": "EPSG:4326", "transform": from_origin(BBOX[0], BBOX[3], RES, RES), "width": w, "height": h}


def stac_client():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def warp_asset(href: str, grid: Dict, resampling=Resampling.bilinear, src_nodata=None) -> np.ndarray:
    """Read a (remote COG) band reprojected onto ``grid``; nodata -> NaN."""
    out = np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_nodata=src.nodata if src_nodata is None else src_nodata, dst_nodata=np.nan,
            dst_transform=grid["transform"], dst_crs=grid["crs"], resampling=resampling,
        )
    return out


def s1_db_for_date(cat, date: str, grid: Dict) -> tuple:
    """Mosaic all orbit-165 RTC frames of one date -> ([H, W, 2] VV/VH dB, item ids)."""
    items = [it for it in cat.search(collections=["sentinel-1-rtc"], bbox=BBOX, datetime=date).items()
             if it.properties.get("sat:relative_orbit") == ORBIT]
    if not items:
        raise RuntimeError(f"No Sentinel-1 RTC orbit {ORBIT} item for {date}")
    out = np.full((grid["height"], grid["width"], 2), np.nan, dtype=np.float32)
    for it in items:
        for b, pol in enumerate(("vv", "vh")):
            lin = warp_asset(it.assets[pol].href, grid)
            with np.errstate(divide="ignore", invalid="ignore"):
                db = 10.0 * np.log10(np.where(lin > 0, lin, np.nan))
            fill = np.isnan(out[..., b]) & np.isfinite(db)
            out[..., b][fill] = db[fill]
    logger.info("S1 %s: %d frame(s), %.1f%% valid", date, len(items), 100 * np.isfinite(out[..., 0]).mean())
    return out, [it.id for it in items]


def s2_item(cat, date: str):
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=date).items())
    # Prefer the tile/orbit with the largest footprint over the AOI (R019 covers it fully).
    items.sort(key=lambda it: -(min(it.bbox[2], BBOX[2]) - max(it.bbox[0], BBOX[0])))
    return items[0] if items else None


def pixel_area_km2(grid: Dict) -> np.ndarray:
    """Per-row pixel area (km^2) on the EPSG:4326 grid."""
    lat = BBOX[3] - (np.arange(grid["height"]) + 0.5) * RES
    return (RES * 111.320 * np.cos(np.deg2rad(lat))) * (RES * 110.574)


def area_km2(mask: np.ndarray, row_area: np.ndarray) -> float:
    return float((mask.sum(axis=1) * row_area).sum())


def agreement(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    a, b = a & valid, b & valid
    inter, union = np.sum(a & b), np.sum(a | b)
    return {
        "iou": float(inter / union) if union else 0.0,
        "frac_sar_water_confirmed": float(inter / max(a.sum(), 1)),
        "frac_optical_water_detected": float(inter / max(b.sum(), 1)),
        "pixel_agreement": float(np.sum((a == b) & valid) / max(valid.sum(), 1)),
        "valid_pixels": int(valid.sum()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    missing = [p for p in CKPTS if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Train the official SAR models first; missing {missing}")

    grid = target_grid()
    row_area = pixel_area_km2(grid)  # [H]
    profile = {"crs": grid["crs"], "transform": grid["transform"], "width": grid["width"], "height": grid["height"]}
    cat = stac_client()
    logger.info("AOI %s -> grid %dx%d px", BBOX, grid["width"], grid["height"])

    # 1-2. Sentinel-1 flood scene and dry-season reference (median of 3 dates).
    flood, flood_ids = s1_db_for_date(cat, FLOOD_DATE, grid)
    ref_stack, ref_ids = [], []
    for d in REF_DATES:
        arr, ids = s1_db_for_date(cat, d, grid)
        ref_stack.append(arr.astype(np.float16))
        ref_ids += ids
    with np.errstate(invalid="ignore"):
        ref = np.nanmedian(np.stack(ref_stack).astype(np.float32), axis=0)
    del ref_stack

    # Domain check against Sen1Floods11 official-train statistics.
    train_stats = json.loads(Path("data/stats/train_stats_official_s1.json").read_text())
    domain = {
        "train_mean_db": train_stats["mean"], "train_std_db": train_stats["std"],
        "kerala_flood_mean_db": [float(np.nanmean(flood[..., b])) for b in range(2)],
        "kerala_flood_std_db": [float(np.nanstd(flood[..., b])) for b in range(2)],
        "kerala_ref_mean_db": [float(np.nanmean(ref[..., b])) for b in range(2)],
    }

    # 3. Predict with the 3-seed SAR ensemble.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [load_model_from_checkpoint(p, device) for p in CKPTS]
    models, norms = [m for m, _, _ in loaded], [n for _, n, _ in loaded]
    probs = {}
    for name, img in (("flood", flood), ("reference", ref)):
        invalid = ~np.all(np.isfinite(img), axis=-1)
        p = predict_tiled(models, img, norms, tile=512, overlap=128, device=device)
        p[invalid] = np.nan
        probs[name] = p
        write_geotiff(OUT / f"kerala_{name}_prob.tif", p, profile, "float32", np.nan)
        logger.info("Predicted %s water: %.1f%% of valid pixels", name, 100 * np.nanmean(p > 0.5))

    valid = np.isfinite(probs["flood"]) & np.isfinite(probs["reference"])
    water_flood = (probs["flood"] > 0.5) & valid
    sar_ref = (probs["reference"] > 0.5) & valid
    # Permanent water (sea, Vembanad Lake) from ESA WorldCover 2021, class 80. Wind-roughened
    # sea can be radar-bright in a reference scene; without this it would count as "flood".
    wc_item = next(cat.search(collections=["esa-worldcover"], bbox=BBOX, datetime="2021").items())
    permanent = warp_asset(wc_item.assets["map"].href, grid, Resampling.nearest, src_nodata=0) == 80
    water_ref = (sar_ref | permanent) & valid
    flooded = water_flood & ~water_ref
    classes = np.full(water_flood.shape, 255, np.uint8)
    classes[valid] = 0
    classes[water_ref] = 1          # permanent / pre-existing water
    classes[flooded] = 2            # flood water
    write_geotiff(OUT / "kerala_flood_classes.tif", classes, profile, "uint8", 255)

    # 5. Independent optical cross-check (Sentinel-2 L2A, 2018-08-22).
    s2 = s2_item(cat, S2_CHECK_DATE)
    b03 = warp_asset(s2.assets["B03"].href, grid) if s2 else None
    b11 = warp_asset(s2.assets["B11"].href, grid) if s2 else None
    scl = warp_asset(s2.assets["SCL"].href, grid, Resampling.nearest, src_nodata=0) if s2 else None
    check = {}
    if s2 is not None:
        clear = np.isin(scl, [4, 5, 6, 7]) & valid & (b03 > 0) & (b11 > 0)  # veg, bare, water, unclassified
        with np.errstate(divide="ignore", invalid="ignore"):
            mndwi = (b03 - b11) / (b03 + b11)
        check = {
            "s2_item": s2.id,
            "s2_tile_cloud_cover_pct": s2.properties.get("eo:cloud_cover"),
            "aoi_clear_fraction": float(clear.sum() / max(valid.sum(), 1)),
            "vs_mndwi": agreement(water_flood, (mndwi > MNDWI_TAU) & clear, clear),
            "vs_scl_water": agreement(water_flood, (scl == 6) & clear, clear),
        }

    peak = s2_item(cat, S2_PEAK_DATE)
    s2_week = [
        {"id": it.id, "date": str(it.datetime.date()), "cloud_cover_pct": round(it.properties["eo:cloud_cover"], 1)}
        for it in cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime="2018-08-14/2018-08-21").items()
    ]

    summary = {
        "aoi_bbox_lonlat": BBOX,
        "grid": {"crs": grid["crs"], "res_deg": RES, "width": grid["width"], "height": grid["height"]},
        "s1_flood_items": flood_ids, "s1_reference_items": ref_ids,
        "models": [str(p) for p in CKPTS],
        "valid_area_km2": area_km2(valid, row_area),
        "water_on_flood_date_km2": area_km2(water_flood, row_area),
        "reference_water_km2": area_km2(water_ref, row_area),
        "reference_water_sar_only_km2": area_km2(sar_ref, row_area),
        "worldcover_permanent_water_km2": area_km2(permanent & valid, row_area),
        "flooded_area_without_worldcover_mask_km2": area_km2(water_flood & ~sar_ref, row_area),
        "worldcover_item": wc_item.id,
        "flooded_area_km2": area_km2(flooded, row_area),
        "flooded_fraction_of_aoi": float(flooded.sum() / max(valid.sum(), 1)),
        "domain_check": domain,
        "optical_cross_check": check,
        "s2_flood_week_scenes": s2_week,
    }
    (RESULTS / "kerala_case_study.json").write_text(json.dumps(summary, indent=2))

    make_figure(flood, classes, peak, grid)
    make_map(classes, grid)
    md = render(summary)
    (RESULTS / "kerala_case_study.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print(md)


def _class_rgba(classes: np.ndarray) -> np.ndarray:
    rgba = np.zeros(classes.shape + (4,), np.uint8)
    for val, hexc in ((1, BLUE), (2, ORANGE)):
        rgb = tuple(int(hexc[i : i + 2], 16) for i in (1, 3, 5))
        rgba[classes == val] = (*rgb, 230)
    return rgba


def make_figure(flood_db: np.ndarray, classes: np.ndarray, peak_item, grid: Dict) -> None:
    ext = (BBOX[0], BBOX[2], BBOX[1], BBOX[3])
    small = {**grid, "width": grid["width"] // 4, "height": grid["height"] // 4,
             "transform": from_origin(BBOX[0], BBOX[3], RES * 4, RES * 4)}
    panels: List = []
    if peak_item is not None:
        tci = np.zeros((small["height"], small["width"], 3), np.float32)
        with rasterio.open(peak_item.assets["visual"].href) as src:
            for b in range(3):
                reproject(rasterio.band(src, b + 1), tci[..., b], src_nodata=0, dst_nodata=0,
                          dst_transform=small["transform"], dst_crs=small["crs"], resampling=Resampling.average)
        panels.append((np.clip(tci / 255.0, 0, 1), None,
                       f"Sentinel-2, {S2_PEAK_DATE}\n(flood peak: {peak_item.properties['eo:cloud_cover']:.0f}% cloud)"))
    vv = np.clip((flood_db[..., 0] + 25) / 25, 0, 1)
    panels.append((vv, "gray", f"Sentinel-1 VV, {FLOOD_DATE}\n(radar sees through cloud)"))
    panels.append((_class_rgba(classes), None, "SAR U-Net flood map\n(3-seed ensemble)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 5.0), facecolor=SURFACE)
    for ax, (im, cmap, title) in zip(axes, panels):
        ax.set_facecolor("#e9e8e4" if im.ndim == 3 and im.shape[-1] == 4 else SURFACE)
        ax.imshow(im, cmap=cmap, extent=ext, interpolation="nearest")
        ax.set_title(title, fontsize=10, loc="left")
        ax.tick_params(labelsize=8, colors=INK2)
        ax.set_xlabel("Longitude (°E)", fontsize=8.5, color=INK2)
    axes[0].set_ylabel("Latitude (°N)", fontsize=8.5, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)]
    fig.legend(handles, ["Reference water (dry season + permanent)", "Flood water (new on 21 Aug 2018)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("August 2018 Kerala floods, Kuttanad / Alappuzha", fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    fig.savefig(FIGS / "kerala_case_study.png", dpi=130)
    plt.close(fig)


def make_map(classes: np.ndarray, grid: Dict) -> None:
    import folium

    step = 2  # ~20 m overlay keeps the HTML small
    rgba = _class_rgba(classes[::step, ::step])
    center = [(BBOX[1] + BBOX[3]) / 2, (BBOX[0] + BBOX[2]) / 2]
    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite imagery",
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        image=rgba, bounds=[[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]], name="FloodSeg SAR flood map (21 Aug 2018)",
        opacity=0.85, mercator_project=True,
    ).add_to(m)
    folium.Rectangle([[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]], color=INK2, weight=1, fill=False,
                     tooltip="Analysis area").add_to(m)
    legend = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fcfcfb;padding:8px 10px;'
        'border:1px solid #d6d5d0;border-radius:4px;font:12px sans-serif;color:#0b0b0b">'
        "<b>FloodSeg - Kerala, 21 Aug 2018</b><br>"
        f'<span style="display:inline-block;width:12px;height:12px;background:{BLUE};margin-right:6px"></span>'
        "Reference water (dry season + permanent)<br>"
        f'<span style="display:inline-block;width:12px;height:12px;background:{ORANGE};margin-right:6px"></span>'
        "Flood water (new)<br><span style=\"color:#52514e\">Sentinel-1 SAR U-Net, 3-seed ensemble</span></div>"
    )
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)
    m.save(str(OUT / "kerala_flood_map.html"))
    logger.info("Wrote %s", OUT / "kerala_flood_map.html")


def render(s: Dict) -> str:
    d, c = s["domain_check"], s["optical_cross_check"]
    L = [
        "# Milestone 5: Case Study - August 2018 Kerala Floods", "",
        f"**Area:** Kuttanad / Alappuzha, {s['aoi_bbox_lonlat']} (lon/lat), "
        f"{s['grid']['width']}×{s['grid']['height']} px at ~10 m (EPSG:4326, Sen1Floods11 grid).  ",
        "**Model:** SAR-only U-Net, 3-seed ensemble trained on the official Sen1Floods11 split.  ",
        f"**Data:** Sentinel-1 RTC from Microsoft Planetary Computer: flood scene {FLOOD_DATE} "
        f"(orbit {ORBIT}, descending); dry-season reference = per-pixel median of {', '.join(REF_DATES)}, "
        "plus ESA WorldCover 2021 permanent water (class 80: sea, Vembanad Lake).", "",
        "## Why SAR here", "", "| Sentinel-2 scene | Tile cloud cover |", "| :--- | :---: |",
    ]
    for sc in s["s2_flood_week_scenes"]:
        L.append(f"| {sc['date']} ({sc['id'][:3]}) | {sc['cloud_cover_pct']}% |")
    L += ["", "Optical imagery during the flood peak week was almost entirely clouded; Sentinel-1 imaged the area regardless.", "",
          "## Flood extent", "", "| Quantity | Value |", "| :--- | :---: |",
          f"| Valid analysis area | {s['valid_area_km2']:.0f} km² |",
          f"| Water on {FLOOD_DATE} | {s['water_on_flood_date_km2']:.0f} km² |",
          f"| Reference water (dry-season SAR ∪ WorldCover permanent water) | {s['reference_water_km2']:.0f} km² |",
          f"| **Flooded (new water)** | **{s['flooded_area_km2']:.0f} km²** ({100 * s['flooded_fraction_of_aoi']:.1f}% of area) |", "",
          f"Without the WorldCover permanent-water mask the flooded area would be "
          f"{s['flooded_area_without_worldcover_mask_km2']:.0f} km²: wind-roughened sea in the dry-season scenes is "
          "radar-bright, is missed as reference water, and would otherwise be counted as flood.", ""]
    if c:
        L += ["## Independent optical cross-check", "",
              f"Sentinel-2 L2A `{c['s2_item']}` (one day after the SAR pass, {c['s2_tile_cloud_cover_pct']:.0f}% tile cloud cover); "
              f"{100 * c['aoi_clear_fraction']:.0f}% of the area is cloud-free per the Scene Classification Layer. "
              "On those pixels only:", "",
              "| Optical reference | IoU with SAR water | SAR water confirmed by optical | Optical water found by SAR | Pixel agreement |",
              "| :--- | :---: | :---: | :---: | :---: |"]
        for key, name in (("vs_mndwi", f"MNDWI > {MNDWI_TAU}"), ("vs_scl_water", "SCL water class")):
            a = c[key]
            L.append(f"| {name} | {a['iou'] * 100:.1f}% | {a['frac_sar_water_confirmed'] * 100:.1f}% | "
                     f"{a['frac_optical_water_detected'] * 100:.1f}% | {a['pixel_agreement'] * 100:.1f}% |")
        L.append("")
    L += ["## Caveats", "",
          f"- **Sensor processing shift:** Planetary Computer supplies radiometrically terrain-corrected gamma0 (RTC); "
          f"Sen1Floods11 uses GEE sigma0 GRD. Scene mean VV/VH here: {d['kerala_flood_mean_db'][0]:.1f} / "
          f"{d['kerala_flood_mean_db'][1]:.1f} dB vs train {d['train_mean_db'][0]:.1f} / {d['train_mean_db'][1]:.1f} dB. "
          "No recalibration is applied.",
          "- **Timing:** the S1 pass (21 Aug) is ~4 days after the peak (~16–17 Aug), so the mapped extent understates peak inundation.",
          "- **Reference water** includes the sea, Vembanad Lake, backwaters and paddy fields that were flooded in the dry season; "
          "'flood water' is the increase over that baseline, not total water.",
          "- **No ground truth:** the optical cross-check is agreement with an independent sensor on clear pixels, "
          "not accuracy. Optical water indices also miss water under vegetation and in cloud shadow.",
          "", "![Kerala case study](../figures/kerala_case_study.png)", "",
          "Interactive map: `outputs/kerala/kerala_flood_map.html` (open in a browser). "
          "GeoTIFFs: `outputs/kerala/kerala_flood_classes.tif` (0 land, 1 reference water, 2 flood, 255 no-data), "
          "`kerala_flood_prob.tif`, `kerala_reference_prob.tif`."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
