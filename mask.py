from resources import *
from experiment_runner import *
from experiment_setup import *

dataset = [join("dataset_ii/", f) for f in listdir("dataset_ii") if isfile(join("dataset_ii/", f))]
d1 = [f for f in dataset if f[-5] == "1"]
print(d1)
model = run_pipeline("dataset_ii/4_left_middle_4_cam1.png", caching=0,
                     mask_method="morph",
                     prealign_method="id",
                     preprocess_method="id",
                     extraction_method="maximum_curvature",
                     postprocess_method="id",
                     postalign_method="id")
