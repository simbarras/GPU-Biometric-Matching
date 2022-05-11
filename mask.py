from random import sample
from experiment_runner import *
from experiment_setup import *

dataset = [join("dataset_ii/", f) for f in listdir("dataset_ii") if isfile(join("dataset_ii/", f))]
d1 = [f for f in dataset if f[-5] == "2"]
d1_s = sample(d1, 100)

a = "dataset_ii/3_left_index_4_cam2.png"
b = "dataset_ii/3_left_index_2_cam2.png"

d1_s = [a, b]
avg = np.zeros((240, 376))

imgs = []
prev = None
for d in d1_s:
    print("extracting...")
    model = run_pipeline(d, caching=False,
                         mask_method="edge",
                         prealign_method="translation",
                         preprocess_method="hist_eq",
                         extraction_method="maximum_curvature",
                         postprocess_method="id",
                         postalign_method="id",
                         model=prev)
    prev = model

    #model[:, :] *= 2
    model[model == 1] = 1
    model[0, 0] = 3
    plt.axis("off")
    plt.imshow(model)
    plt.show()
    #avg += model
    imgs.append(model)
    #plt.imshow(model)
    #plt.show()
    #plt.show()


#imgs_2 = []
#for d in d1_s:
#    print("extracting...")
#    model = run_pipeline(d, caching=False,
#                     mask_method="edge",
#                     prealign_method="translation",
#                     preprocess_method="hist_eq",
#                     extraction_method="maximum_curvature",
#                     postprocess_method="id",
#                     postalign_method="erode_com")
#    #avg += model
#    imgs_2.append(model)

#plt.show()

#compute_single_distance(imgs[0], imgs[1], "random_subsampling_dist")
compute_single_distance(imgs[0], imgs[1], "miura_distance")
# compute_single_distance(imgs_2[0], imgs_2[1], "miura_distance")

#compute_single_distance(imgs[0], imgs[1], "skeleton_hd")