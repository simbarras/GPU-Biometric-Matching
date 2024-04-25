#!/usr/bin/env python3

from pathlib import *
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from resources import *
import time


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
    print(f'\r({i}/{len(files)}) Processing {file}...          ', end='')
    veins = run_pipeline(file,
                         mask_method="edge",
                         prealign_method="translation",
                         preprocess_method="id",
                         extraction_method="maximum_curvature",
                         postprocess_method="id",
                         postalign_method="miura_matching")
    pipelinedImages.append((Path(file).stem, veins))

print("\rRunning pipeline finished.                                      ")
print("Results have been computed for:") 

distSame = []
distDiff = []

for (i, (f1, v1)) in enumerate(pipelinedImages):

    print(f'\r({i}/{len(pipelinedImages)}) Processing {f1}...          ', end='')

    cam1 = f1[-1]
    fileIdentifier1 = f1[0:13]

    distSameImage = []
    distDiffImage = []

    for (f2, v2) in pipelinedImages[i:]:
        cam2 = f2[-1]
        fileIdentifier2 = f2[0:13]

        if cam1 == cam2:
            veins2 = postalign(v2, "miura_matching", v1)
            dist = compute_miura_distance(v1, veins2)

            if fileIdentifier1 == fileIdentifier2:
                distSameImage.append(dist)
                continue
            
            distDiffImage.append(dist)

    distSame.append((f1, distSameImage))
    distDiff.append((f1, distDiffImage))

with open("distance_results/distances_different_finger.csv", "w") as diffFile:
    with open("distance_results/distances_same_finger.csv", "w") as sameFile:
        for (n, dists) in distSame:
            sameFile.write(n + ", ")

            for d in dists:
                sameFile.write(str(d) + ", ")
            sameFile.write("\n")
        
        for (n, dists) in distDiff:
            diffFile.write(n + ", ")

            for d in dists:
                diffFile.write(str(d) + ", ")
            diffFile.write("\n")