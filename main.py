# requirements
import os
import ray
import math
import time
import json
import signal
import requests
import rasterio
import subprocess
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from datetime import datetime, timedelta


# global variables
china = gpd.read_file("china.geojson")
grids = china.copy()
hospitals_path = "hospitals.csv"
hospitals = pd.read_csv(hospitals_path)
hospitals_gdf = gpd.GeoDataFrame(hospitals, geometry=gpd.points_from_xy(hospitals["lng"], hospitals["lat"]))
grids_id = ""
hospitals_id = ""
S1_batch_size = 70
S2_batch_size = 200


# cost in osm
def try4distance(start, end):
    base = "http://localhost:8081/dijkstra"
    headers = {"Content-Type": "application/json"}
    data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": []}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": []}},
        ],
    }
    data["features"][0]["geometry"]["coordinates"] = start
    data["features"][1]["geometry"]["coordinates"] = end
    response = requests.post(base, data=json.dumps(data), headers=headers).json()
    cost = response["features"][0]["properties"]["weight"]
    return cost


# ---GA2SFCA---

# ---Gussian---
def gussian_w(t):
    if t > 60:
        return 0
    return (math.exp(-1 / 2 * (t / 60) ** 2) - math.exp(-1 / 2)) / (1 - math.exp(-1 / 2))

# S1, process a grid
@ray.remote
def process_grid(grid, start):
    centroid = grid.geometry.centroid
    end = [centroid.x, centroid.y]
    cost = try4distance(start, end)
    if cost == "no path found":
        return 0
    cost = float(cost)

    t = grid["sum"]
    t = 0 if math.isnan(t) else t
    PW = 0 if t == 0 else t * gussian_w(cost)
    return 0 if PW == 0 else PW

# S1, process a hospital
@ray.remote
def process_hospital_i(i, hospital, length):
    hospital["R"] = 0.0
    S = hospital["beds"]
    S = 5 if math.isnan(S) else S
    start = [hospital["lng"], hospital["lat"]]

    # find the nearest grids
    global grids_id
    buffer = hospital.geometry.buffer(1.2)
    grids = ray.get(grids_id)
    filtered_grids = grids[grids.within(buffer)]

    result_handles = [
        process_grid.remote(filtered_grids.iloc[j], start)
        for j in range(len(filtered_grids))
    ]
    for handle in result_handles:
        hospital["R"] += ray.get(handle)

    if hospital["R"] != 0:
        hospital["R"] = S / hospital["R"] * 10000

    if i % 100 == 0:
        print_and_log(f"S1, {i}/{length}")

    return f"{i},{hospital['R']}\n"

# S1, reindex and merge
def reindex_S1():
    hospitals = pd.read_csv("./data/tmp/grids_hospitals.csv")
    S1Rj = pd.read_csv("./data/tmp/S1Rj.csv")
    S1Rj = S1Rj.sort_values(by="id")
    for i in range(len(S1Rj)):
        index = S1Rj.loc[i, "id"]
        row = hospitals.iloc[index]
        S1Rj.at[i, "id"] = row["id"]
    # merge by id
    S1Rj = pd.merge(S1Rj, hospitals, on="id")
    S1Rj.to_csv("./data/tmp/S1Rj_reindex.csv", index=False)

# S2, process a hospital
@ray.remote
def process_hospital(hospital, start):
    end = [hospital["lng"], hospital["lat"]]
    cost = try4distance(start, end)
    if cost == "no path found":
        return 0
    cost = float(cost)
    RW = hospital["R"] * gussian_w(cost)
    return 0 if RW == 0 else RW

# S2, process a grid
@ray.remote
def process_grid_i(i, grid, length):
    grid["acc"] = 0.0
    centroid = grid.geometry.centroid
    start = [centroid.x, centroid.y]

    # find the nearest hospitals
    buffer = grid.geometry.buffer(1.2)
    hospitals_gdf = ray.get(hospitals_id)
    filtered_hospitals = hospitals_gdf[hospitals_gdf.within(buffer)]

    result_handles = [
        process_hospital.remote(filtered_hospitals.iloc[j], start)
        for j in range(len(filtered_hospitals))
    ]
    for handle in result_handles:
        grid["acc"] += ray.get(handle)

    if i % 10000 == 0:
        print_and_log(f"S2, {i}/{length}")

    return f"{i},{grid['acc']}\n"

# S2, reindex and merge
def reindex_acc(region_name, grids):
    acc = pd.read_csv("./data/tmp/acc.csv")
    for i in range(len(acc)):
        index = acc.loc[i, "index"]
        row = grids.iloc[index]
        acc.at[i, "index"] = row["index"]

    acc = acc.sort_values(by="index")
    acc.to_csv("./data/tmp/acc_reindex.csv", index=False)

    grids = pd.merge(grids, acc, left_on="index", right_on="index")
    grids.to_file(f"./data/result/{region_name}acc.geojson", driver="GeoJSON")


# process a region
def main(region_name, shape):
    # --- init ---
    global grids
    global grids_id
    global hospitals
    global hospitals_id
    global hospitals_gdf
    global hospitals_path

    global S1_batch_size
    global S2_batch_size

    check_and_remove("./data/tmp/S1Rj.csv")
    check_and_remove("./data/tmp/acc.csv")
    check_and_remove("./data/tmp/S1Rj_reindex.csv")
    check_and_remove("./data/tmp/acc_reindex.csv")

    check_and_create("./data/tmp/S1Rj.csv", "id,R\n")
    check_and_create("./data/tmp/acc.csv", "index,acc\n")

    hospitals = pd.read_csv(hospitals_path)
    hospitals_gdf = gpd.GeoDataFrame(hospitals, geometry=gpd.points_from_xy(hospitals["lng"], hospitals["lat"]))
    hospitals_gdf = gpd.clip(hospitals_gdf, shape)
    hospitals_gdf.to_csv("./data/tmp/grids_hospitals.csv", index=False)
    # --- init ---

    # --- Step 1 ---
    print_and_log("Step 1 start...")
    time1 = time.time()
    grids_id = ray.put(grids)

    hospitals_length = len(hospitals_gdf)
    batches = [
        hospitals_gdf.iloc[i : i + S1_batch_size]
        for i in range(0, hospitals_length, S1_batch_size)
    ]
    exists = 0
    for batch in batches:
        res = ray.get(
            [
                process_hospital_i.remote(i + exists, batch.iloc[i], hospitals_length)
                for i in range(len(batch))
            ]
        )
        with open("./data/tmp/S1Rj.csv", "a") as f:
            for r in res:
                f.write(r)
        exists += len(batch)

    reindex_S1()
    print_and_log(f"Step 1 finished, in {time.time()-time1} s.")
    # --- Step 1 ---

    hospitals = pd.read_csv("./data/tmp/S1Rj_reindex.csv")
    hospitals_gdf = gpd.GeoDataFrame(hospitals, geometry=gpd.points_from_xy(hospitals["lng"], hospitals["lat"]))
    hospitals_id = ray.put(hospitals_gdf)

    # --- Step 2 ---
    print_and_log("Step 2 start ...")
    time2 = time.time()

    grids_length = len(grids)
    batches = [
        grids.iloc[i : i + S2_batch_size] 
        for i in range(0, grids_length, S2_batch_size)
    ]
    exists = 0
    for batch in batches:
        res = ray.get(
            [
                process_grid_i.remote(i + exists, batch.iloc[i], grids_length)
                for i in range(len(batch))
            ]
        )
        with open("./data/tmp/acc.csv", "a") as f:
            for r in res:
                f.write(r)
        exists += len(batch)

    reindex_acc(region_name, grids)
    print_and_log(f"Step 2 finished, in {time.time()-time2} s.")
    # --- Step 2 ---

    print_and_log(f"total: {time.time() - time1} s.")
    print_and_log("----------------")

    # --- clean ---
    check_and_remove(f"./data/tmp/{region_name}.tif")
    check_and_remove(f"./data/tmp/{region_name}.geojson")
    check_and_remove(f"./data/tmp/{region_name}_roads.osm.pbf")
    check_and_remove(f"./data/tmp/{region_name}_roads.osm.pbf.fmi")
    # --- clean ---


def get_pop_in_gdf(gdf, rn):
    print_and_log(f"getting pop in the region: {rn}...")
    tif_path = f"chn_ppp_2020_1km.tif"
    pop_path = f"./data/tmp/{rn}.tif"

    with rasterio.open(tif_path) as src:
        out_image, out_transform = mask(src, gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        with rasterio.open(pop_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print_and_log(f"raster to shape")
    subprocess.run(
        f"gdal_polygonize.py {pop_path} -b 1 -f 'GeoJSON' -overwrite ./data/tmp/{rn}.geojson OUTPUT sum",
        shell=True,
    )

    print_and_log(f"pop finished, fill index")
    grids = gpd.read_file(f"./data/tmp/{rn}.geojson")
    grids["index"] = range(len(grids))
    grids.to_file(f"./data/tmp/{rn}.geojson", driver="GeoJSON")

    return grids


def convert_into_pbf(path):
    output = path + ".pbf"
    print_and_log(f"packing pbf ...")
    subprocess.run(["osmium", "cat", path, "-o", output])
    print_and_log(f"packing pbf finished")
    return output


def get_roads_in_pbf(gdf, name):
    print_and_log(f"getting roads in region ...")
    bounds = gdf.total_bounds
    north, south, east, west = bounds[3], bounds[1], bounds[2], bounds[0]
    path = f"./data/tmp/{name}_roads.osm.pbf"
    subprocess.run(
        [
            "osmconvert",
            "china-latest.osm.pbf",
            f"-b={west},{south},{east},{north}",
            f"-o={path}",
        ]
    )
    print_and_log(f"roads finished")
    return path


# print to console and log to file
def print_and_log(s):
    now_utc = datetime.utcnow()
    now = now_utc + timedelta(hours=8)
    now = now.strftime("%Y-%m-%d %H:%M:%S")
    s = f"[{now}] {s}"
    print(s)
    with open("./data/tmp/logs.txt", "a") as f:
        f.write(s + "\n")


def check_and_remove(path):
    if os.path.exists(path):
        os.remove(path)


def check_and_create(path, head):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(head)


check_and_create("./data/tmp/logs.txt", "")

for I, R in china.iloc[::-1].iterrows():
    region = gpd.GeoDataFrame([R])
    region = region.drop(columns=["adchar"])
    rn = region["name"].values[0]

    if os.path.exists(f"./data/result/{rn}acc.geojson"):
        print_and_log(f"{rn} already exists in the result")
        continue

    print_and_log(f"processing: {rn}")

    # get roads in the region
    pbf = ""
    if not os.path.exists(f"./data/tmp/{rn}_roads.osm.pbf"):
        pbf = get_roads_in_pbf(region, rn)
    else:
        pbf = f"./data/tmp/{rn}_roads.osm.pbf"
        print_and_log(f"{rn} roads already exist")

    # get pop in the region
    time1 = time.time()
    if not os.path.exists(f"./data/tmp/{rn}.geojson"):
        region = get_pop_in_gdf(region, rn)
    else:
        region = gpd.read_file(f"./data/tmp/{rn}.geojson")
        print_and_log(f"{rn}范围内人口已存在")
    grids = gpd.read_file(f"./data/tmp/{rn}.geojson")
    print_and_log(f"划分格网用时: {time.time()-time1} s")

    # preprocess the road network
    if not os.path.exists(f"./data/tmp/{rn}_roads.osm.pbf.fmi"):
        print_and_log("preprocessing road network ...")
        p = subprocess.run(f"taskset -c 0,1 cargo run --release -p osm_ch_pre {pbf}", shell=True)
    else:
        print_and_log(f"{rn} preprocessed road network already exists")

    # run the service
    print_and_log("run the service ...")
    p = subprocess.Popen(f"cargo run --release -p osm_ch_web {pbf}.fmi", shell=True, preexec_fn=os.setsid)
    time.sleep(30)

    try:
        ray.init()
        main(rn, gpd.GeoDataFrame([R]))
    finally:
        p.terminate()
        p.wait()
        os.killpg(p.pid, signal.SIGTERM)
        ray.shutdown()
