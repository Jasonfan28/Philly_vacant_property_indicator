"""
Recompute docs/ward_stats.json from docs/vacancy_predictions.pmtiles.

Adds the `observed_vacant` field per ward by walking maxzoom (z=15) tiles,
decoding MVT, deduping parcels by parcel_number, and aggregating
sum(ovs == 1) by geographic_ward. Existing fields (total, flagged,
mean_prob) are recomputed from the same source so everything is
internally consistent.

Run:  python docs/build_ward_stats.py
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import mapbox_vector_tile
from pmtiles.reader import MmapSource, Reader

PMTILES_PATH = Path(__file__).parent / "vacancy_predictions.pmtiles"
OUT_PATH     = Path(__file__).parent / "ward_stats.json"
LAYER        = "parcels"


def lonlat_to_xyz(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def main() -> None:
    with open(PMTILES_PATH, "rb") as fh:
        src = MmapSource(fh)
        reader = Reader(src)
        header = reader.header()
        z      = header["max_zoom"]
        min_lon = header["min_lon_e7"] / 1e7
        max_lon = header["max_lon_e7"] / 1e7
        min_lat = header["min_lat_e7"] / 1e7
        max_lat = header["max_lat_e7"] / 1e7

        x_min, y_max = lonlat_to_xyz(min_lon, min_lat, z)
        x_max, y_min = lonlat_to_xyz(max_lon, max_lat, z)
        candidates = (x_max - x_min + 1) * (y_max - y_min + 1)
        print(f"Scanning z={z} tiles in [{x_min}-{x_max}] × [{y_min}-{y_max}] "
              f"({candidates} candidates)")

        per_ward_total        = defaultdict(int)
        per_ward_flagged      = defaultdict(int)
        per_ward_observed     = defaultdict(int)
        per_ward_underreport  = defaultdict(int)
        per_ward_prob_sum     = defaultdict(float)
        seen: set[str] = set()
        tiles_with_data = 0

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                blob = reader.get(z, x, y)
                if not blob:
                    continue
                tiles_with_data += 1
                # pmtiles stores tile data raw; tile_compression tells us if gzipped
                if header["tile_compression"].name == "GZIP":
                    blob = gzip.decompress(blob)
                decoded = mapbox_vector_tile.decode(blob)
                layer = decoded.get(LAYER)
                if not layer:
                    continue
                for feat in layer["features"]:
                    props = feat["properties"]
                    pn = props.get("parcel_number")
                    if not pn or pn in seen:
                        continue
                    seen.add(pn)
                    ward = props.get("geographic_ward")
                    if ward is None:
                        continue
                    ward = int(ward)
                    flag = int(props.get("ensemble_flag") or 0)
                    ovs  = int(props.get("ovs") or 0)
                    per_ward_total[ward]       += 1
                    per_ward_flagged[ward]     += flag
                    per_ward_observed[ward]    += 1 if ovs == 1 else 0
                    per_ward_underreport[ward] += 1 if (flag == 1 and ovs == 0) else 0
                    per_ward_prob_sum[ward]    += float(props.get("ensemble_prob") or 0)

        print(f"Tiles with data: {tiles_with_data}")
        print(f"Unique parcels:  {len(seen)}")

    rows = []
    for ward in sorted(per_ward_total.keys()):
        total = per_ward_total[ward]
        rows.append({
            "ward":            ward,
            "total":           total,
            "flagged":         per_ward_flagged[ward],
            "mean_prob":       per_ward_prob_sum[ward] / total if total else 0.0,
            "observed_vacant": per_ward_observed[ward],
            "underreported":   per_ward_underreport[ward],
        })

    OUT_PATH.write_text(json.dumps(rows, indent=2))
    grand_total        = sum(r["total"] for r in rows)
    grand_flagged      = sum(r["flagged"] for r in rows)
    grand_observed     = sum(r["observed_vacant"] for r in rows)
    grand_underreport  = sum(r["underreported"] for r in rows)
    print(f"Wards: {len(rows)}  parcels: {grand_total}  "
          f"flagged: {grand_flagged}  observed_vacant: {grand_observed}  "
          f"underreported: {grand_underreport}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
