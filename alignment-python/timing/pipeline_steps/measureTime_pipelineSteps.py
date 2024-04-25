#!/usr/bin/env python3

from pathlib import *
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from resources import *
import time


width = 376
height = 240

files = []
dataset_path = Path('../dataset')
for file in dataset_path.glob('*.png'):
    files.append(str(file))

files.sort()

print("Warm-up started.")

for i in range(0, 100):
    model = run_pipeline(files[0],
                         mask_method="edge",
                         prealign_method="translation",
                         preprocess_method="id",
                         extraction_method="maximum_curvature",
                         postprocess_method="id",
                         postalign_method="miura_matching")
    
print("Warm-up finished.\nTiming for pipeline steps done for:")

timesMask = []
timesPreal = []
timesMCurv = []

for i, file in enumerate(files):
    print(f'\r({i}/{len(files)}) Processing {file}...          ', end='')

    timesMaskPerImage = []
    timesPrealPerImage = []
    timesMCurvPerImage = []

    cam = int(file[-5])
    img_orig = Image.open(file)
    mask = None
    img2 = None
    mask2 = None

    for j in range(0, 15):
        img = np.asarray(img_orig)
        timeStart = time.time_ns()
        img, mask = extract_mask(img, cam, "edge")
        timeEnd = time.time_ns()
        duration = (timeEnd - timeStart) / 1000000000
        timesMaskPerImage.append(duration)
    
    timesMask.append((Path(file).stem, timesMaskPerImage))

    for j in range(0, 15):
        timeStart = time.time_ns()
        img2, mask2 = prealign(img, mask, "translation", cam)
        timeEnd = time.time_ns()
        duration = (timeEnd - timeStart) / 1000000000
        timesPrealPerImage.append(duration)
    
    timesPreal.append((Path(file).stem, timesPrealPerImage))

    for j in range(0, 15):
        timeStart = time.time_ns()
        img, mask = extract_features(img2, mask2, "maximum_curvature")
        timeEnd = time.time_ns()
        duration = (timeEnd - timeStart) / 1000000000
        timesMCurvPerImage.append(duration)
    
    timesMCurv.append((Path(file).stem, timesMCurvPerImage))

with open("timing/pipeline_steps/edge_mask/edge_mask_time.csv", "w") as edgeMFile:
    for (n, timings) in timesMask:
        edgeMFile.write(n + ", ")

        for t in timings:
            edgeMFile.write(str(t) + ", ")
        edgeMFile.write("\n")

with open("timing/pipeline_steps/prealignment/prealignment_time.csv", "w") as preFile:
    for (n, timings) in timesPreal:
        preFile.write(n + ", ")

        for t in timings:
            preFile.write(str(t) + ", ")
        preFile.write("\n")

with open("timing/pipeline_steps/maximum_curvature/maxCurv_time.csv", "w") as maxCFile:
    for (n, timings) in timesMCurv:
        maxCFile.write(n + ", ")

        for t in timings:
            maxCFile.write(str(t) + ", ")
        maxCFile.write("\n")
        