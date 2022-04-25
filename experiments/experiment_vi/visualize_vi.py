from visualize import *
plt.style.use('ggplot')

labels = ["i - huang fingertip - same", "ii - miura - same", "iii - id - same", "iv - huang fingertip - different", "v - huang + fingertip - same", "vi - CoM - same", "vii - miura - same", "viii - id - different", "ix - miura - different"]

show_histogram(["i", "ii", "iii", "iv"], labels)

eer_i, tpr_i, fpr_i = get_eer_confusion("i", "iv", "similarity")
eer_ii, tpr_ii, fpr_ii = get_eer_confusion("ii", "iv", "similarity")
eer_iii, tpr_iii, fpr_iii = get_eer_confusion("iii", "iv", "similarity")

print("Different Finger")
print(eer_i, eer_ii, eer_iii)

show_roc([tpr_i, tpr_ii, tpr_iii],
         [fpr_i, fpr_ii, fpr_iii],
         ["i -  Huang Leftmost EER: " + str(round(eer_i, 3)),
          "ii -  Miura Matching EER=" + str(round(eer_ii, 3)),
          "iii - Identity EER=" + str(round(eer_iii, 3))],
         title="ROC vs. Different Finger Identity")