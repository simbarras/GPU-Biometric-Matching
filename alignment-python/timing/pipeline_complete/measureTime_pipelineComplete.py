#!/usr/bin/env python3

from pathlib import *
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from resources import *
import time

if len(sys.argv) != 3:
    print("Invalid number of arguments. Please provide min and max.")
    exit(1)

start_from = int(sys.argv[1])
end_before = int(sys.argv[2])


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
    
print("Warm-up finished.\nComplete pipeline timing done for:")

times = []

for i, file in enumerate(files):
    if i < start_from or i >= end_before:
        continue

    print(f'({i}/{len(files)}) Processing {file}...          ', end='\n')

    timesPerImage = []

    for j in range(0, 15):
        timeStart = time.time_ns()
        veins = run_pipeline(file,
                            mask_method="edge",
                            prealign_method="translation",
                            preprocess_method="id",
                            extraction_method="maximum_curvature",
                            postprocess_method="id",
                            postalign_method="miura_matching")
        timeEnd = time.time_ns()
        duration = (timeEnd - timeStart) / 1000000000
        timesPerImage.append(duration)
    
    times.append((Path(file).stem, timesPerImage))

with open("timing/pipeline_complete/pipeline_complete.csv", "w") as pipComplete:
    for (n, timings) in times:
        pipComplete.write(n + ", ")

        for t in timings:
            pipComplete.write(str(t) + ", ")
        pipComplete.write("\n")
        