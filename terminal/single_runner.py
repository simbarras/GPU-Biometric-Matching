from resources import *
import sys
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

import json
from os.path import isfile, join
from pathlib import Path
from pytictoc import TicToc

######################## globals
threshold = 0.88 # IDK about the threshold i might play around it
# or just ask prof about it
# I think the distance is inverted, so the lower the distance the better the match
 
def match_veins(model_path_png, probe_path_png):
        ###################################################################### Run extraction pipeline
        model = run_pipeline(model_path_png,
                             cam=0,
                             mask_method="edge",
                             prealign_method="translation",
                             preprocess_method="id",
                             extraction_method="maximum_curvature",
                             postprocess_method="id",
                             postalign_method="miura_matching")

        probe = run_pipeline(probe_path_png,
                             cam=0,
                             mask_method="edge",
                             prealign_method="translation",
                             preprocess_method="id",
                             extraction_method="maximum_curvature",
                             postprocess_method="id",
                             postalign_method="miura_matching",
                             model=model)

        ###################################################################### Compute distance
        distance = compute_single_distance(model, probe, "miura_distance")

        if distance < threshold:
            return 1
        else:
            return 0

if len(sys.argv) != 3:
    print("Usage: python3 single_runner.py <model_file> <probe_file>")
    exit(1)
    

model_file = sys.argv[1]
probe_file = sys.argv[2]

res = match_veins(model_file, probe_file)

print(res)