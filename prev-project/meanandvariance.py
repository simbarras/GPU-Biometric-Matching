import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom
import math, random, scipy
from matplotlib.backends.backend_pdf import PdfPages

USER = 1
FINGER = 2
CAM = 3
LEFTRIGHT = 4
ATTEMPT = 5
'''this script computes the mean and variance for both hashes coming from the same person and coming from different persons
an entropy estimation is also calculated

one should compute and store the hashes beforehand -- see hashing.py in resources

'''


def compute_mean_var(data):
    # print(np.std(np.asarray(data)),np.std(np.asarray(data))**2 ==np.var(np.asarray(data))  )
    return np.mean(np.asarray(data)), np.var(np.asarray(data)) # var = mean(x), where x = abs(a - a.mean())**2.

def compute_hamming_dist(data):
    res_list = []
    for (i,j) in data:
        h1 = np.load(i)
        h2 = np.load(j)

        prod = h1*h2 # elmt prod -1*-1 gives 1 same as 1*1; rest gives -1//or 0
        len_prod = len(prod)
        nr_of_ones = np.count_nonzero(prod == 1)
        # print("nr of 1s in prod: ",nr_of_ones)
        anything_but_1 = len_prod - nr_of_ones
        different_pixels = anything_but_1/len_prod
        # test = scipy.spatial.distance.hamming(h1,h2)
        # print(different_pixels,test)
        # print("ratio of difrt: ", different_pixels)
        res_list.append(different_pixels)
        # res_list.append(anything_but_1)
        # print(np.count_nonzero(h1 == 1)/3500,np.count_nonzero(h1 == -1)/3500)
        # print(np.count_nonzero(h2 == 1)/3500,np.count_nonzero(h2 == -1)/3500)
    return res_list

def compute_hamming_dist_subsample(data,substr_size):
    res_list = []
    positions = np.array(random.SystemRandom().sample(range(0,3500), substr_size))
    for (i,j) in data:
        h1 = np.load(i)
        h2 = np.load(j)
        v1 = np.array([h1[y] for y in positions])
        v2 = np.array([h2[y] for y in positions])

        prod = v1*v2 # elmt prod -1*-1 gives 1 same as 1*1; rest gives -1//or 0
        len_prod = len(prod)
        nr_of_ones = np.count_nonzero(prod == 1)
        # print("nr of 1s in prod: ",nr_of_ones)
        anything_but_1 = len_prod - nr_of_ones
        different_pixels = anything_but_1/len_prod
        # print("ratio of difrt: ", different_pixels)
        res_list.append(different_pixels)
    return res_list

def estimate_entropy(mu,var,len):
    # in case we can approx w bin distr (yet this is an asspt), this is entropy of binomial
    N = (mu * (1 - mu))/var #deg of frdm
    q = 1 - mu
    H = -mu * math.log2(mu) - q * math.log2(q)
    Ht = H*N
    # print(N,H,Ht)
    rate = Ht / len
    # print(entropy_rate)
    return N,H,Ht,rate

def estimate_avg_entropy_subsamples(data):
    results = []
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        iter = []
        for t in range(0,10):
            difrt_hd_k = compute_hamming_dist_subsample(data,k)
            m,v = compute_mean_var(difrt_hd_k)
            N,H,Ht,rate = estimate_entropy(m,v,k)
            # results.append((k,Ht,rate))
            iter.append(rate)
        results.append((k,np.mean(np.asarray(iter))))
    return results

def generate_data_to_compare():
    possibilities = []
    for user in ['5','3']:
        for finger in ['middle','ring', 'index']:
            for cam in ['cam1','cam2']:
                for lr in ['left','right']:
                    img_trial1 = "data/HsH_" + user + "_" + finger + "_" + cam + "_" + lr + "_1.npy"
                    img_trial2 = "data/HsH_" + user + "_" + finger + "_" + cam + "_" + lr + "_2.npy"
                    if img_trial1 != "data/HsH_5_middle_cam1_left_1.npy" and img_trial1 != "data/HsH_5_middle_cam2_left_1.npy" and img_trial1 !=  "data/HsH_5_index_cam1_left_1.npy" and img_trial1 !=  "data/HsH_5_index_cam2_left_1.npy":
                        possibilities.append(img_trial1)
                        possibilities.append(img_trial2)

    same_finger_and_cam_differnt_attempt = []
    different_finger = []
    for i in possibilities:
        split_i = i.split('_')
        for j in possibilities:
            split_j =  j.split('_')
            # print(split_j)
            if split_j[LEFTRIGHT]==split_i[LEFTRIGHT] and split_j[USER]==split_i[USER] and split_j[FINGER] == split_i[FINGER]:
                if split_j[CAM] == split_i[CAM] and split_j[ATTEMPT]!= split_i[ATTEMPT]:
                    if (i,j) not in same_finger_and_cam_differnt_attempt and (j,i) not in same_finger_and_cam_differnt_attempt:
                        same_finger_and_cam_differnt_attempt.append((i,j))
                # print("don't want to compare with same image, different picture taken nor different camera ")
            else:
                if (i,j) not in different_finger and (j,i) not in different_finger: #no duplicates
                    different_finger.append((i,j))
    return same_finger_and_cam_differnt_attempt,different_finger

# the distribution plot is taken from BOB codebase
def plot_distributions(HDs, p, sigma, N):
	""" Plots the histogram of the input Hamming Distances (HDs) and fits a binomial to the distribution.

	**Parameters:**

	HDs (list): A list of Hamming distances between the fingervein patterns of every pair of different fingers
    p (float): The mean of the HD distribution and the fitted binomial distribution
	sigma (float): The standard deviation of the HD distribution and the fitted binomial distribution
	N (float): The number of degrees of freedom (approx. number of independent bits in the underlying fingervein feature vectors) for the fitted binomial distribution

	**Returns:**

	HD_distribution.pdf (saved image): Plot of the HD distribution with the corresponding binomial distribution overlaid

	"""

	# Calculate the histogram of the Hamming distance (HD) distribution:
	fig = plt.figure()
	plt.xlim([0, 1])
	plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
	# plt.title('whatever')
	plt.xlabel('HD')
	plt.ylabel('Probability')
	counts, bin_edges = np.histogram(HDs, bins=np.arange(0.000, 1.001, 0.001))  # bin the HD values
	bin_probs = counts/float(len(HDs))  # normalise the bin counts so that every bin value gives the probability of that bin
	bin_middles = (bin_edges[1:]+bin_edges[:-1])/float(2)  # get the mid point of each bin
	bin_width = bin_edges[1]-bin_edges[0]  # compute the bin width
	# Calculate the binomial distribution corresponding to the HD histogram, and plot the HD histogram overlaid with the binomial distribution:
	N = int(round(N))  # degrees of freedom for the binomial distribution = degrees of freedom of HD distribution
	x_temp = range(0, N+1)
	x = [i/float(N) for i in x_temp] # fractional Hamming distances from 1/N to N/N (liken to probability of getting H = x * N heads from N coin tosses)
	plt.plot(x, scipy.stats.binom.pmf(x_temp, N, p), '-k')
	plt.ylim([0.0, 0.10])
	plt.yticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
	# Normalize the histogram so it is easier to visualise the binomial distribution fit:
	norm_scale = max(scipy.stats.binom.pmf(x_temp, N, p))/max(bin_probs)
	normed_bin_probs = bin_probs * norm_scale
	plt.bar(bin_middles, normed_bin_probs, width=bin_width, color='lightgrey')
	# Save the figure:
	fig_path = 'data/' + '/HD_distribution.pdf'
	pdf = PdfPages(fig_path)
	fig.savefig(fig_path, format='pdf', bbox_inches='tight')
	plt.close()

'''main'''

same, different = generate_data_to_compare()
# print(len(same),len(different))
same_hd =  compute_hamming_dist(same)
difrt_hd = compute_hamming_dist(different)

# Creating histogram
fig, ax = plt.subplots(1, 1, tight_layout=True)
ax.hist(difrt_hd)
# Show plot
# plt.show()
m_same, var_same = compute_mean_var(same_hd)
print("mean, var same:", m_same, var_same)
m_difrt, var_difrt = compute_mean_var(difrt_hd)
print("mean, var different:", m_difrt, var_difrt )
# print(estimate_entropy(m_difrt,var_difrt,3500))
N,H,Ht,rate = estimate_entropy(m_difrt,var_difrt,3500)
print(N,H,Ht,rate)
# print(estimate_avg_entropy_subsamples(different))

plot_distributions(difrt_hd, m_difrt, math.sqrt(var_difrt), N)
