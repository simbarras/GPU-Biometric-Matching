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

print("Run pipeline for each image.")

pipelinedImages = []

for i, file in enumerate(files):
    
    print(f'({i}/{len(files)}) Processing {file}...          ', end='\n')
    veins = run_pipeline(file,
                         mask_method="edge",
                         prealign_method="translation",
                         preprocess_method="id",
                         extraction_method="maximum_curvature",
                         postprocess_method="id",
                         postalign_method="miura_matching")
    pipelinedImages.append((Path(file).stem, veins))

print("Running pipeline finished.                                      ")
print("Results have been computed for:") 

timesPost = []
timesDist = []

for (i, (f1, v1)) in enumerate(pipelinedImages):

    if i < start_from or i >= end_before:
        continue

    print(f'({i}/{len(pipelinedImages)}) Processing {f1}...          ', end='\n')

    cam1 = f1[-1]
    fileIdentifier1 = f1[0:13]

    timesPostPerImage = []
    timesDistPerImage = []

    for (f2, v2) in pipelinedImages[i:]:
        cam2 = f2[-1]
        fileIdentifier2 = f2[0:13]

        if cam1 == cam2:

            timeSpanPost = 0

            veins2 = postalign(v2, "miura_matching", v1)
            for j in range(0, 10):
                timeStart = time.time_ns()
                veins2 = postalign(v2, "miura_matching", v1)
                timeEnd = time.time_ns()
                duration = (timeEnd - timeStart) / 1000000000
                timeSpanPost += duration
            
            timesPostPerImage.append(timeSpanPost / 10)

            timeSpanDist = 0

            for j in range(0, 10):
                timeStart = time.time_ns()
                dist = compute_miura_distance(v1, veins2)
                timeEnd = time.time_ns()
                duration = (timeEnd - timeStart) / 1000000000
                timeSpanDist += duration

            timesDistPerImage.append(timeSpanDist / 10)

    timesPost.append((f1, timesPostPerImage))
    timesDist.append((f1, timesDistPerImage))

with open("timing/pipeline_steps/postalignment/postalignment_time.csv", "w") as postFile:
    with open("timing/pipeline_steps/distance/distance_time.csv", "w") as distFile:
        for (n, times) in timesPost:
            postFile.write(n + ", ")

            for t in times:
                postFile.write(str(t) + ", ")
            postFile.write("\n")
        
        for (n, times) in timesDist:
            distFile.write(n + ", ")

            for t in times:
                distFile.write(str(t) + ", ")
            distFile.write("\n")