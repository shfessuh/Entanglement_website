import json
import os
import pandas as pd
import numpy as np
from shapely.geometry import shape, mapping
from shapely.affinity import translate, scale
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

# ================================================================
# 1. LOAD + PREP THE POLICY DATA
# ================================================================

CSV = "FinalVariable_CountyPolicies_DATASET2018_state_and_county_0314.csv"

df = pd.read_csv(CSV, dtype={"FIPS": str, "StateFIPS": str})
df["FIPS"] = df["FIPS"].str.zfill(5)
df["StateFIPS"] = df["StateFIPS"].str.zfill(2)
df["Year"] = df["Year"].astype(int)

YEARS = sorted(df["Year"].unique().tolist())          
YEAR_IDX = {y: i for i, y in enumerate(YEARS)}

COUNTY_TYPES = {
    "county_lt_Anti-Solicitation": "Anti-Solicitation",
    "county_lt_E- Verify":         "E-Verify",
    "county_lt_Language Access":   "Language Access",
    "county_lt_Secure Communities": "Secure Communities",
}
STATE_TYPES = {
    "state_lt_Anti-Solicitation":       "Anti-Solicitation",
    "state_lt_E- Verify":               "E-Verify",
    "state_lt_Language Access":         "Language Access",
    "state_lt_REALID Act":              "REAL ID Act",
    "state_lt_REALID Act + E- Verify":  "REAL ID Act + E-Verify",
}
def unfav_types(frame, level, type_map):
    unfav = frame[f"{level}_unfavorable"]
    fav = frame[f"{level}_favorable"]
    out = {}
    for col, label in type_map.items():
        out[label] = np.where((unfav > 0) & (fav == 0), frame[col], 0.0)
    return pd.DataFrame(out, index=frame.index)

cty_t = unfav_types(df, "county", COUNTY_TYPES)
st_t = unfav_types(df, "state", STATE_TYPES)

m = (df["State"] == "Colorado") & (df["Year"] == 2008)
st_t.loc[m, "E-Verify"] = 1.0

m = (df["FIPS"] == "53011") & (df["Year"] == 2009)
cty_t.loc[m, "E-Verify"] = 1.0

# sanity: types must sum to the unfavorable count on every row
assert (cty_t.sum(axis=1) == df["county_unfavorable"]).all()
assert (st_t.sum(axis=1) == df["state_unfavorable"]).all()


def year_arrays(sub_df, count_col, type_frame):
    """Return (Count, cum, typesCount, typesCum) arrays for one jurisdiction."""
    Count = [0.0] * len(YEARS)
    t_Count = {lab: [0.0] * len(YEARS) for lab in type_frame.columns}
    for i, row in sub_df.iterrows():
        yi = YEAR_IDX[row["Year"]]
        Count[yi] = float(row[count_col])
        for lab in type_frame.columns:
            t_Count[lab][yi] = float(type_frame.loc[i, lab])
    cum = np.cumsum(Count).tolist()
    t_cum = {lab: np.cumsum(v).tolist() for lab, v in t_Count.items()}
    # drop all-zero type arrays
    t_Count = {k: v for k, v in t_Count.items() if any(v)}
    t_cum = {k: v for k, v in t_cum.items() if any(v)}
    return Count, cum, t_Count, t_cum


def to_int_if_whole(arr):
    return [int(v) if float(v).is_integer() else round(v, 4) for v in arr]


policy = {
    "years": YEARS,
    "countyTypeOrder": list(COUNTY_TYPES.values()),
    "stateTypeOrder": list(STATE_TYPES.values()),
    "county": {},
    "state": {},
}

# ---- county level ----
for fips, sub in df.groupby("FIPS"):
    if sub["county_unfavorable"].sum() == 0:
        continue  
    Count, cum, t_Count, t_cum = year_arrays(sub, "county_unfavorable", cty_t.loc[sub.index])
    policy["county"][fips] = {
        "Count": to_int_if_whole(Count),
        "cum": to_int_if_whole(cum),
        "typesCount": {k: to_int_if_whole(v) for k, v in t_Count.items()},
        "typesCum": {k: to_int_if_whole(v) for k, v in t_cum.items()},
    }

state_rows = df.drop_duplicates(subset=["StateFIPS", "Year"])
for sfips, sub in state_rows.groupby("StateFIPS"):
    if sub["state_unfavorable"].sum() == 0:
        continue
    Count, cum, t_Count, t_cum = year_arrays(sub, "state_unfavorable", st_t.loc[sub.index])
    policy["state"][sfips] = {
        "name": sub["State"].iloc[0],
        "Count": to_int_if_whole(Count),
        "cum": to_int_if_whole(cum),
        "typesCount": {k: to_int_if_whole(v) for k, v in t_Count.items()},
        "typesCum": {k: to_int_if_whole(v) for k, v in t_cum.items()},
    }

policy["maxima"] = {
    "county": {
        "Count": max((max(v["Count"]) for v in policy["county"].values()), default=0),
        "cum": max((max(v["cum"]) for v in policy["county"].values()), default=0),
    },
    "state": {
        "Count": max((max(v["Count"]) for v in policy["state"].values()), default=0),
        "cum": max((max(v["cum"]) for v in policy["state"].values()), default=0),
    },
}

with open("policy_data.json", "w") as f:
    json.dump(policy, f, separators=(",", ":"))

print(f"policy_data.json: {len(policy['county'])} counties, "
      f"{len(policy['state'])} states with unfavorable policies")
print("maxima:", policy["maxima"])

# ================================================================
# 3. GEOMETRY: mainland + Alaska/Hawaii insets (same as old project)
#    Static props only - FIPS, StateFIPS, names. Policy values are
#    looked up in policy_data.json client-side, so the geojson
#    doesn't change when toggles change.
# ================================================================

with open("georef-united-states-of-america-county.json") as f:
    geo = json.load(f)

counties = []
for item in geo:
    geom_raw = shape(item["geo_shape"]["geometry"])
    geom_simplified = geom_raw.simplify(0.01, preserve_topology=True)
    counties.append({
        "FIPS": str(item["coty_code"][0]).zfill(5),
        "ste_name": item["ste_name"][0],
        "coty_name": item["coty_name"][0],
        "geometry": geom_simplified,
        "geometry_raw": geom_raw,
    })
del geo

territories = ["Puerto Rico", "American Samoa", "United States Virgin Islands",
               "Guam", "Commonwealth of the Northern Mariana Islands"]

mainland = [c for c in counties
            if c["ste_name"] not in territories and not c["FIPS"].startswith(("02", "15"))]
alaska = [c for c in counties if c["FIPS"].startswith("02")]
hawaii = [c for c in counties if c["FIPS"].startswith("15")]

clip_box = shapely_box(-170, 50, -129, 72)
for c in alaska:
    c["geometry"] = c["geometry"].intersection(clip_box)
    c["geometry_raw"] = c["geometry_raw"].intersection(clip_box)
alaska = [c for c in alaska if not c["geometry"].is_empty]

ak_union = unary_union([c["geometry"] for c in alaska])
ak_cx, ak_cy = ak_union.centroid.x, ak_union.centroid.y
for c in alaska:
    c["geometry"] = translate(
        scale(c["geometry"], xfact=0.35, yfact=0.35, origin=(ak_cx, ak_cy)),
        xoff=35, yoff=-40,
    )
    c["geometry_raw"] = translate(
        scale(c["geometry_raw"], xfact=0.35, yfact=0.35, origin=(ak_cx, ak_cy)),
        xoff=35, yoff=-40,
    )

ak_xs = [c["geometry"].bounds[0] for c in alaska] + [c["geometry"].bounds[2] for c in alaska]
ak_ys = [c["geometry"].bounds[1] for c in alaska] + [c["geometry"].bounds[3] for c in alaska]
ak_bounds = [min(ak_xs), min(ak_ys), max(ak_xs), max(ak_ys)]

hi_clip = shapely_box(-161, 18.5, -154.5, 22.5)
for c in hawaii:
    c["geometry"] = c["geometry"].intersection(hi_clip)
    c["geometry_raw"] = c["geometry_raw"].intersection(hi_clip)
hawaii = [c for c in hawaii if not c["geometry"].is_empty]

hi_union = unary_union([c["geometry"] for c in hawaii])
hi_cx, hi_cy = hi_union.centroid.x, hi_union.centroid.y
for c in hawaii:
    c["geometry"] = scale(c["geometry"], xfact=1.5, yfact=1.5, origin=(hi_cx, hi_cy))
    c["geometry_raw"] = scale(c["geometry_raw"], xfact=1.5, yfact=1.5, origin=(hi_cx, hi_cy))

hi_xs = [c["geometry"].bounds[0] for c in hawaii] + [c["geometry"].bounds[2] for c in hawaii]
hi_ys = [c["geometry"].bounds[1] for c in hawaii] + [c["geometry"].bounds[3] for c in hawaii]
hi_bounds_raw = [min(hi_xs), min(hi_ys), max(hi_xs), max(hi_ys)]

x_shift = (ak_bounds[2] + 0.3) - hi_bounds_raw[0]
y_shift = (ak_bounds[1] + ak_bounds[3]) / 2 - (hi_bounds_raw[1] + hi_bounds_raw[3]) / 2
for c in hawaii:
    c["geometry"] = translate(c["geometry"], xoff=x_shift, yoff=y_shift)
    c["geometry_raw"] = translate(c["geometry_raw"], xoff=x_shift, yoff=y_shift)

all_counties = mainland + alaska + hawaii

geojson_features = []
for c in all_counties:
    geojson_features.append({
        "type": "Feature",
        "properties": {
            "FIPS": c["FIPS"],
            "StateFIPS": c["FIPS"][:2],
            "name": c["coty_name"],
            "state": c["ste_name"],
        },
        "geometry": mapping(c["geometry"]),
    })

geojson = {"type": "FeatureCollection", "features": geojson_features}

def round_coords(coords, precision=4):
    if isinstance(coords[0], (list, tuple)):
        return [round_coords(cc, precision) for cc in coords]
    return [round(cc, precision) for cc in coords]

for feat in geojson["features"]:
    g = feat["geometry"]
    if g["type"] in ("Polygon", "MultiPolygon"):
        g["coordinates"] = round_coords(g["coordinates"])

with open("counties.geojson", "w") as f:
    json.dump(geojson, f, separators=(",", ":"))

from collections import defaultdict
by_state = defaultdict(list)
state_names = {}
for c in all_counties:
    sfips = c["FIPS"][:2]
    by_state[sfips].append(c["geometry_raw"])
    state_names[sfips] = c["ste_name"]

state_features = []
for sfips, geoms in by_state.items():
    merged = unary_union(geoms)
    merged = merged.simplify(0.01, preserve_topology=True)
    state_features.append({
        "type": "Feature",
        "properties": {"StateFIPS": sfips, "state": state_names[sfips]},
        "geometry": mapping(merged),
    })

states_geojson = {"type": "FeatureCollection", "features": state_features}
for feat in states_geojson["features"]:
    g = feat["geometry"]
    if g["type"] in ("Polygon", "MultiPolygon"):
        g["coordinates"] = round_coords(g["coordinates"])

with open("states.geojson", "w") as f:
    json.dump(states_geojson, f, separators=(",", ":"))
print(f"states.geojson: {len(state_features)} states")

hi_bounds = [min(c["geometry"].bounds[0] for c in hawaii),
             min(c["geometry"].bounds[1] for c in hawaii),
             max(c["geometry"].bounds[2] for c in hawaii),
             max(c["geometry"].bounds[3] for c in hawaii)]
policy["insetBounds"] = {"akBounds": ak_bounds, "hiBounds": hi_bounds, "pad": 0.9}
with open("policy_data.json", "w") as f:
    json.dump(policy, f, separators=(",", ":"))

geo_mb = os.path.getsize("counties.geojson") / 1024 / 1024
pol_kb = os.path.getsize("policy_data.json") / 1024
print(f"counties.geojson: {len(geojson_features)} features, {geo_mb:.1f} MB")
print(f"policy_data.json: {pol_kb:.0f} KB")