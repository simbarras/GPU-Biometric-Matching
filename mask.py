from random import sample

from resources import *
from experiment_runner import *
from experiment_setup import *

dataset = [join("dataset_ii/", f) for f in listdir("dataset_ii") if isfile(join("dataset_ii/", f))]
d1 = [f for f in dataset if f[-5] == "2"]

d1_s = sample(d1, 100)

avg = np.zeros((240, 376))
for d in d1_s:
    model = run_pipeline(d, caching=0,
                     mask_method="edge",
                     prealign_method="id",
                     preprocess_method="id",
                     extraction_method="maximum_curvature",
                     postprocess_method="id",
                     postalign_method="id")

    avg += model
    plt.imshow(avg)
    plt.show()