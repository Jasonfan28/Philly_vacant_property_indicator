"""
Build per-area stats and fetch boundary polygons.

In one pass over docs/vacancy_predictions.pmtiles we aggregate per
ward / census tract / zip code, then write three companion stats files:

    docs/ward_stats.json    (ward, total, flagged, mean_prob,
                             observed_vacant, underreported)
    docs/tract_stats.json   (tract, ...)
    docs/zip_stats.json     (zip,   ...)

We also download the tract and ZIP polygon GeoJSONs from OpenDataPhilly
(the same source as docs/ward_boundaries.geojson) if they are not
already present:

    docs/tract_boundaries.geojson
    docs/zip_boundaries.geojson

Run:  python docs/build_area_stats.py
"""

from __future__ import annotations

import gzip
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path

import mapbox_vector_tile
from pmtiles.reader import MmapSource, Reader

DOCS = Path(__file__).parent
PMTILES_PATH = DOCS / "vacancy_predictions.pmtiles"
LAYER        = "parcels"

WARD_OUT  = DOCS / "ward_stats.json"
TRACT_OUT = DOCS / "tract_stats.json"
ZIP_OUT   = DOCS / "zip_stats.json"

TRACT_GEO = DOCS / "tract_boundaries.geojson"
ZIP_GEO   = DOCS / "zip_boundaries.geojson"

# OpenDataPhilly ArcGIS feature services — same provider as ward_boundaries.geojson
TRACT_URL = (
    "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/"
    "Census_Tracts_2010/FeatureServer/0/query"
    "?where=1=1&outFields=GEOID10,NAME10&outSR=4326&f=geojson"
)
ZIP_URL = (
    "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/"
    "Zipcodes_Poly/FeatureServer/0/query"
    "?where=1=1&outFields=code&outSR=4326&f=geojson"
)


def lonlat_to_xyz(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def aggregate() -> tuple[dict, dict, dict, int]:
    with open(PMTILES_PATH, "rb") as fh:
        reader = Reader(MmapSource(fh))
        header = reader.header()
        z = header["max_zoom"]
        x_min, y_max = lonlat_to_xyz(header["min_lon_e7"] / 1e7, header["min_lat_e7"] / 1e7, z)
        x_max, y_min = lonlat_to_xyz(header["max_lon_e7"] / 1e7, header["max_lat_e7"] / 1e7, z)

        wards  = defaultdict(lambda: dict(total=0, flagged=0, observed=0, underreport=0, prob_sum=0.0))
        tracts = defaultdict(lambda: dict(total=0, flagged=0, observed=0, underreport=0, prob_sum=0.0))
        zips   = defaultdict(lambda: dict(total=0, flagged=0, observed=0, underreport=0, prob_sum=0.0))

        seen: set[str] = set()
        gz = header["tile_compression"].name == "GZIP"

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                blob = reader.get(z, x, y)
                if not blob:
                    continue
                if gz:
                    blob = gzip.decompress(blob)
                layer = mapbox_vector_tile.decode(blob).get(LAYER)
                if not layer:
                    continue
                for feat in layer["features"]:
                    p = feat["properties"]
                    pn = p.get("parcel_number")
                    if not pn or pn in seen:
                        continue
                    seen.add(pn)
                    flag = int(p.get("ensemble_flag") or 0)
                    ovs  = int(p.get("ovs") or 0)
                    prob = float(p.get("ensemble_prob") or 0)
                    under = 1 if (flag == 1 and ovs == 0) else 0

                    for bucket, key in (
                        (wards,  p.get("geographic_ward")),
                        (tracts, p.get("census_tract")),
                        (zips,   p.get("zip_code")),
                    ):
                        if key is None:
                            continue
                        b = bucket[int(key)]
                        b["total"]       += 1
                        b["flagged"]     += flag
                        b["observed"]    += 1 if ovs == 1 else 0
                        b["underreport"] += under
                        b["prob_sum"]    += prob

    return wards, tracts, zips, len(seen)


def to_rows(bucket: dict, area_key: str) -> list[dict]:
    rows = []
    for area_id in sorted(bucket.keys()):
        b = bucket[area_id]
        total = b["total"]
        rows.append({
            area_key:         area_id,
            "total":           total,
            "flagged":         b["flagged"],
            "mean_prob":       b["prob_sum"] / total if total else 0.0,
            "observed_vacant": b["observed"],
            "underreported":   b["underreport"],
        })
    return rows


def fetch_geojson(url: str, dst: Path, label: str) -> None:
    if dst.exists():
        print(f"  {label}: already present at {dst.name}")
        return
    print(f"  {label}: downloading…")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    # ArcGIS sometimes paginates with `exceededTransferLimit` — Philly tracts/zips
    # are well under the 2000-record default, so a single request suffices.
    parsed = json.loads(data)
    feat_count = len(parsed.get("features", []))
    if parsed.get("exceededTransferLimit"):
        raise RuntimeError(f"{label}: transfer limit exceeded; pagination not implemented")
    dst.write_bytes(data)
    print(f"  {label}: wrote {dst.name} ({feat_count} features, {len(data)//1024} KB)")


def main() -> None:
    print("Aggregating PMTiles…")
    wards, tracts, zips, parcel_count = aggregate()
    print(f"  unique parcels: {parcel_count}")

    ward_rows  = to_rows(wards,  "ward")
    tract_rows = to_rows(tracts, "tract")
    zip_rows   = to_rows(zips,   "zip")

    WARD_OUT.write_text(json.dumps(ward_rows,  indent=2))
    TRACT_OUT.write_text(json.dumps(tract_rows, indent=2))
    ZIP_OUT.write_text(json.dumps(zip_rows,    indent=2))
    print(f"  wards: {len(ward_rows)}   tracts: {len(tract_rows)}   zips: {len(zip_rows)}")
    print(f"  wrote {WARD_OUT.name}, {TRACT_OUT.name}, {ZIP_OUT.name}")

    print("Ensuring boundary files…")
    fetch_geojson(TRACT_URL, TRACT_GEO, "tracts")
    fetch_geojson(ZIP_URL,   ZIP_GEO,   "zips")
    print("Done.")


if __name__ == "__main__":
    main()
