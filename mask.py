from random import sample

from resources import *
from experiment_runner import *
from experiment_setup import *

dataset = [join("dataset_ii/", f) for f in listdir("dataset_ii") if isfile(join("dataset_ii/", f))]
d1 = [f for f in dataset if f[-5] == "1"]
d1_s = sample(d1, 100)

a = "dataset_ii/1_left_index_4_cam1.png"
b = "dataset_ii/13_left_index_5_cam1.png"

d1_s = [a, b]
avg = np.zeros((240, 376))

imgs = []
for d in d1_s:
    print("extracting...")
    model = run_pipeline(d, caching=False,
                     mask_method="fingerfocus",
                     prealign_method="id",
                     preprocess_method="id",
                     extraction_method="maximum_curvature",
                     postprocess_method="skeletonize",
                     postalign_method="id")
    #avg += model
    imgs.append(model)
    plt.imshow(model)
    plt.show()
compute_single_distance(imgs[0], imgs[1], "random_subsampling_dist")
compute_single_distance(imgs[0], imgs[1], "hamming_distance")
compute_single_distance(imgs[0], imgs[1], "skeleton_hd")