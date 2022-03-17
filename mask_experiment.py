from alignment_evaluation import *
from resources import regiongrow
from resources import remove_static_mask
model_path = 'dataset_i/maximum_curvature/5_left_ring_1_cam1'
probe_path = 'dataset_i/maximum_curvature/5_left_ring_2_cam1'

model_img_path = 'dataset_i/5_left_ring_1_cam1.png'
probe_img_path = 'dataset_i/5_left_ring_2_cam1.png'

model_img = Image.open(model_img_path)
probe_img = Image.open(probe_img_path)
# plt.imshow(probe_img)
# plt.show()
W = np.asarray(probe_img)
# W, mask = regiongrow(W, roi=(40, 210, 10, 360))

W = remove_static_mask(W, 1)
W, mask = fingerfocus(W, roi=(40, 210, 40, 360))

plt.imshow(W)
# plt.imshow(mask, alpha=0.5)
plt.show()