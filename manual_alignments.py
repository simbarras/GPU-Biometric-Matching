# This script tries to manually run all kinds of different alignments and combinations thereof, including masks etc.
import matplotlib.pyplot as plt
from skimage.transform import probabilistic_hough_line
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage.morphology import skeletonize
from skimage import data
from matplotlib import cm
from alignment_evaluation import *
from skimage import data, color
from resources.background import *
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from scipy import signal
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

from resources import remove_static_mask
# visualize steps:

def manual_pre_align(img, mask):
    return img, mask

erosion_kernel = [
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 1, 0, 0],
     [0, 0, 1, 1, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0]
]
def rot_score(fv, angle, kernel):
    fv = si.binary_dilation(fv.copy(), [[0, 1, 0], [1, 1, 1], [0, 1, 0]], iterations=2)
    fv = fv.astype("float")
    hor = si.rotate(fv, angle, reshape=False)
    # hor = sp.fftconvolve(fv, np.rot90(kernel, k = 2), 'same')

    #look at top k peaks, take average.
    # extrema = signal.argrelextrema(hor, np.greater)
    # print(extrema)

    # mn = np.min(hor)
    # mx = np.max(hor)
    # thresh = mx - (mx - mn) / 2
    # hor[hor < thresh] = 0
    # a = np.sum(hor, axis=1)


    b_scores = np.zeros((fv.shape[0]))
    b_centers = np.zeros((fv.shape[0]), dtype="uint16")
    score = 0
    center_sum = 0
    vein_len = 0
    # score: look for longest vein in each row -> sum up its values
    for i in range(hor.shape[0]):
        for j in range(hor.shape[1]):
            if hor[i, j] > 0:
                score += hor[i, j]
                vein_len += 1
                center_sum += j
            else:
                if score > b_scores[i]:
                    b_scores[i] = score
                    b_centers[i] = round(center_sum / vein_len)
                score = 0
                center_sum = 0
                vein_len = 0

    # plt.imshow(fv)
    # plt.imshow(hor, cmap="gray", alpha=.4)
    # plt.title("Max score: " + str(np.max(b_scores)))
    # plt.show()
    return hor, b_scores, np.max(b_scores), b_centers

# idea: find strongest vein in finger, align according to this.
# for this, we need to consider different rotations of the image and sum up the pixels horizontally.
def prevalent_vein(fv, mask, y_window=80, sigma_y=3, sigma_x=10, filter_size=40):

    # f = np.fft.fft2(fv)
    # fshift = np.fft.fftshift(f)
    # mag = 20 * np.log(np.abs(fshift))
    # mag[mag < 120] = 0
    # plt.imshow(np.fft.ifft2(f))
    # print(np.max(mag))
    # plt.imshow(mag)
    # plt.show()

    kernel = np.outer(signal.windows.gaussian(filter_size, sigma_y), signal.windows.gaussian(filter_size, sigma_x))
    # kernel = np.ones((1, 40))
    y_half = round(fv.shape[0] / 2)
    x_half = round(fv.shape[1] / 2)

    step_size = 2
    max_angle = 20
    b_hor = fv
    b_score = 0
    b_angle = 0
    b_a = []
    b_centers = []
    for angle in range(-max_angle, max_angle, step_size):
        hor, a, score, centers = rot_score(fv, angle, kernel)
        if score > b_score:
            b_hor = hor
            b_score = score
            b_a = a
            b_angle = angle
            b_centers = centers
    window_start = y_half - round(y_window / 2)
    b_ys = np.argmax(b_a[window_start : window_start + y_window]) + window_start
    b_xs = b_centers[b_ys]
    fv = si.rotate(fv.astype("float"), b_angle, reshape=False, mode='nearest')
    fv = np.rint(fv)
    print(b_ys)

    plt.imshow(fv)
    l = np.zeros(fv.shape)
    l[b_ys, :] = 1
    plt.imshow(l, cmap="gray", alpha=.2)
    plt.imshow(b_hor, alpha=.5)
    plt.show()
    t0 = y_half - b_ys
    s0 = x_half - b_xs
    fv = shift(fv, -t0, -s0)

    return fv, mask


def prob_hough_transform(fv, mask):
    # Line finding using the Probabilistic Hough Transform
    image = fv.astype('uint8') * 255
    # edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(image, threshold=2, line_length=5,
                                     line_gap=2)

    ###### Circles
    hough_radii = np.arange(16, 30, 2)
    hough_res = hough_circle(image, hough_radii)
    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               min_xdistance=5, min_ydistance=5, total_num_peaks=2)
    # Draw them
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
        image[circy, circx] = (220, 20, 20)



    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(image, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(image * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()

def hough_transform(fv, mask):


    # Constructing test image
    image = np.zeros((200, 200))
    idx = np.arange(25, 175)
    image[idx, idx] = 255
    image[line(45, 25, 25, 175)] = 255
    image[line(25, 135, 175, 155)] = 255
    image = fv.astype('uint8')
    plt.imshow(image)
    plt.show()

    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)


    h, theta, d = hough_line(image, theta=tested_angles)
    print(h, len(h))
    print(theta, len(theta))
    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]
    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for _, angle, dist in zip(*(hough_line_peaks(h, theta, d, min_distance=0, min_angle=0, num_peaks=10))):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

    plt.tight_layout()
    plt.show()
    return fv, mask


def process_fv(fv, min_area=10):

    # closing
    fv = si.binary_closing(fv)
    # plt.imshow(fv)
    # plt.show()


    # apply kernel
    kernel = np.outer(signal.windows.gaussian(3, 1), signal.windows.gaussian(3, 1))
    hor = sp.fftconvolve(fv, np.rot90(kernel, k = 2), 'same')
    mx, mn = np.max(hor), np.min(hor)
    thresh = mx - (mx - mn) / 1.1
    hor[hor < thresh] = 0
    hor[hor > 0] = 1
    fv = hor.astype('uint8')
    # plt.imshow(fv)
    # plt.show()

    # skeletonize
    fv = skeletonize(fv)
    # plt.imshow(fv)
    # plt.show()

    # remove noise
    blobs, labnbr = si.label(fv, structure = np.array([[1, 1, 1],
                                                             [1, 1, 1],
                                                             [1, 1, 1]]))
    pixels = blobs.ravel()
    areas = np.bincount(pixels)[1:]
    kept_labels = np.argwhere(areas > min_area) + 1
    fv = np.isin(blobs, kept_labels)
    # plt.imshow(fv)
    # plt.show()
    return fv

def manual_post_align(fv, mask):
    fv = process_fv(fv)
    fv = si.binary_dilation(fv.copy(), [[0, 1, 0], [1, 1, 1], [0, 1, 0]], iterations=2)
    # center = [round(np.average(indices)) for indices in np.where(fv > 0)]
    # y_half = round(fv.shape[0] / 2)
    # x_half = round(fv.shape[1] / 2)
    # plt.imshow(fv)
    # plt.show()
    # fv = shift(fv, center[0] - y_half, center[1]- x_half)
    # plt.imshow(fv)
    # plt.show()
    # fv, mask = prevalent_vein(fv, mask)
    # prob_hough_transform(fv, mask)
    return fv, mask


def manual_post_align_old(fv, mask):
    plt.imshow(fv)
    plt.show()
    # kernel = [
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 1, 0, 0, 0],
    #     [0, 0, 1, 1, 1, 1, 0, 0],
    #     [0, 0, 1, 1, 1, 1, 0, 0],
    #     [0, 0, 0, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0]
    # ]
    # kernel = np.ones((3, 30))

    kernel = np.outer(signal.windows.gaussian(32, 3), signal.windows.gaussian(32, 9))
    plt.imshow(kernel)
    plt.show()
    # hor = si.binary_erosion(fv.copy(), kernel)
    hor = sp.fftconvolve(fv, np.rot90(kernel, k = 2), 'same')
    mn = np.min(hor)
    mx = np.max(hor)
    thresh = mx - (mx - mn) / 2
    hor[hor < thresh] = 0

    # x_window = 100
    y_window = 50
    y_half = round(fv.shape[0] / 2)
    x_half = round(fv.shape[1] / 2)

    # hor[:y_half - round(y_window / 2), :] = 0
    # hor[y_half + round(y_window / 2) :, :] = 0

    lower = y_half - round(y_window / 2)
    a = np.sum(hor[y_half - round(y_window / 2) : y_half + round(y_window / 2), :], axis=1)
    y_t = np.argmax(a) + lower
    t0 = y_half - y_t
    # x_t_l = 0
    # while hor[y_t, x_t_l] == 0:
    #     x_t_l += 1
    # x_t_r = fv.shape[1] - 1
    # print(x_t_r)
    # while hor[y_t, x_t_r] == 0:
    #     x_t_r -= 1
    #
    # x_t =  round((x_t_l + x_t_r) / 2)


    # shift y_t to nearest modulo of 60:
    #y_u, y_d = (-y_t % 60), - (y_t % 60)
    ## # take smallest distance:
    #if abs(y_u) > abs(y_d):
    #    t0 = y_d
    #else:
    #    t0 = y_u

    # hor[:y_t - round(x_window / 2), :] = 0
    # hor[y_t + round(x_window / 2) :, :] = 0


    plt.plot(a)
    plt.show()
    b = hor[y_t, :]#np.sum(hor, axis=0)
    plt.plot(b)
    plt.show()
    # mx = np.max(b)
    # b[b < mx / 4] = 0
    sm = np.sum(b)
    com = 0
    for i in range(b.shape[0]):
        com += i * (b[i] / sm)

    # b = si.gaussian_filter1d(b, 10)
    x_t = round(com)
    s0 = x_half - x_t
    plt.plot(b)
    plt.show()
    print("y: ", y_t)
    print("x: ", x_t)

    # # hor_10 = sp.fftconvolve(fv, np.rot90(kernel_10, k = 2), 'same')
    # # hor_m10 = sp.fftconvolve(fv, np.rot90(kernel_m10, k = 2), 'same')
    # #
    # # a = np.max(hor)
    # # b = np.max(hor_10)
    # # c = np.max(hor_m10)
    # # print(a, b, c)
    # # if a < b:
    # #     hor = hor_10
    # # if b < c:
    # #     hor = hor_m10
    # #
    # t0, s0 = np.unravel_index(hor.argmax(), hor.shape)
    # #
    # # hor[t0, s0] = 1000
    # # plt.imshow(hor)
    # # plt.show()
    # #
    # grid_len = 60
    # # round to grid:
    # t_u, t_d, s_u, s_d = (-t0 % grid_len), - (t0 % grid_len), (-s0 % grid_len), -(s0 % grid_len)
    # # take smallest distance:
    # if abs(t_u) > abs(t_d):
    #     t = t_d
    # else:
    #     t = t_u
    # if abs(s_u) > abs(s_d):
    #     s = s_d
    # else:
    #     s = s_u
    #
    # print(t0, s0)
    # print(t_u, t_d, s_u, s_d)
    # print(t, s)
    print("t0:", t0)
    plt.imshow(fv)
    l = np.zeros(fv.shape)
    l[y_t, :] = 1
    plt.imshow(l, cmap="gray", alpha=.2)
    plt.imshow(hor, alpha=.5)
    plt.show()
    fv = shift(fv, -t0, s0)
    return fv, mask



########################################################################################################################


# parameters:
visualize = [False, False, True]
mask = "dilation"
pre_alignment = "id" # manual, id, leftmost_edge, huang_normalization, huang_fingertip, huang_leftmost
post_alignment = "manual" # manual, id, miura_matching, center_of_mass
score = "hamming_dist_sub_blur" # hamming_dist_sub_blur, hamming_distance

model_img_path = 'dataset_ii/3_right_middle_1_cam2.png'
probe_img_path = 'dataset_ii/18_right_middle_4_cam2.png'

model_img = Image.open(model_img_path)
probe_img = Image.open(probe_img_path)

model = np.asarray(model_img)
probe = np.asarray(probe_img)

# mask
if mask == "dilation":
    cam = int(model_img_path[-5])
    model_mask = dilation_mask(model, cam)
    # model[model_mask == 0] = 0
    cam = int(probe_img_path[-5])
    probe_mask = dilation_mask(probe, cam)
    # probe[probe_mask == 0] = 0

else:
    model, model_mask = fingerfocus(model, roi=(40, 190, 10, 360))
    probe, probe_mask = fingerfocus(probe, roi=(40, 190, 10, 360))
if visualize[0]:
    plt.imshow(model)
    plt.imshow(model_mask, alpha=0.4)
    plt.title("Model with mask (" + mask + ")")
    plt.show()
    plt.imshow(probe)
    plt.imshow(probe_mask, alpha=0.4)
    plt.title("Probe with mask (" + mask + ")")
    plt.show()

# feature extraction + pre-alignment
nomask = np.ones(shape=(240, 376), dtype="uint16")
if pre_alignment == "manual":
    model, model_mask = manual_pre_align(model, model_mask)
    probe, probe_mask = manual_pre_align(probe, probe_mask)
fv_model, model_mask = extract_features(model, model_mask, pre_alignment)
fv_probe, probe_mask = extract_features(probe, probe_mask, pre_alignment)
#fv_model[model_mask == 0] = 0
#fv_probe[probe_mask == 0] = 0
if visualize[1]:
    plt.imshow(fv_model)
    plt.imshow(model, alpha=0.4)
    plt.title("Model pre-aligned + extracted (" + pre_alignment + ")")
    plt.show()
    plt.imshow(fv_probe)
    plt.imshow(probe, alpha=0.4)
    plt.title("Probe pre-aligned + extracted (" + pre_alignment + ")")
    plt.show()

# post alignment
if post_alignment == "manual":
    model, model_mask = manual_post_align(fv_model, model_mask)
    probe, probe_mask = manual_post_align(fv_probe, probe_mask)
else:
    model, model_mask, probe, probe_mask = postprocess(fv_model, model_mask, fv_probe, probe_mask, post_alignment)
if visualize[2]:
    plt.imshow(fv_model)
    plt.title("Model post-aligned (" + post_alignment + ")")
    plt.show()
    plt.imshow(fv_probe)
    plt.title("Probe post-aligned (" + post_alignment + ")")
    plt.show()

# score
plt.title("mask: " + mask + "\nalignment: (" + pre_alignment + ", " + post_alignment + ")")
compute_single_score(model, model_mask, probe, probe_mask, score)