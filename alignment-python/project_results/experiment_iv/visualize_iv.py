from visualize import *
plt.style.use('ggplot')

labels = ["i - id - same", "ii - leftmost - same", "iii - huang - same", "iv - huang + leftmost - same",
          "v - huang + fingertip - same", "vi - CoM - same", "vii - miura - same",
          "viii - id - different", "ix - leftmost - different", "x - huang - different", "xi - huang + leftmost - different",
          "xii - huang + fingertip - different", "xiii - CoM - different", "xiv - miura - different"]

show_histogram(["i", "viii", "xiv"], labels)


cam = None
eer_i, tpr_i, fpr_i = get_eer_confusion("i", "viii", "distance", cam=cam)
eer_ii, tpr_ii, fpr_ii = get_eer_confusion("ii", "ix", "distance", cam=cam)
eer_iii, tpr_iii, fpr_iii = get_eer_confusion("iii", "x", "distance", cam=cam)
eer_iv, tpr_iv, fpr_iv = get_eer_confusion("iv", "xi", "distance", cam=cam)
eer_v, tpr_v, fpr_v = get_eer_confusion("v", "xii", "distance", cam=cam)
eer_vi, tpr_vi, fpr_vi = get_eer_confusion("vi", "xiii", "distance", cam=cam)
eer_vii, tpr_vii, fpr_vii = get_eer_confusion("vii", "xiv", "distance", cam=cam)

print("EERs:")
print(eer_i, eer_ii, eer_iii, eer_iv, eer_v, eer_vi, eer_vii)
show_roc([tpr_i, tpr_ii, tpr_iii, tpr_iv, tpr_v, tpr_vi, tpr_vii, tpr_vii],
         [fpr_i, fpr_ii, fpr_iii, fpr_iv, fpr_v, fpr_vi, fpr_vii, fpr_vii],
         ["i -  Ident. EER=" + str(round(eer_i, 3)),
          "ii -  Leftm. EER=" + str(round(eer_ii, 3)),
          "iii - Huang EER=" + str(round(eer_iii, 3)),
          "iv - H + L EER=" + str(round(eer_iv, 3)),
          "v -  H + F EER=" + str(round(eer_v, 3)),
          "vi -  CoM EER=" + str(round(eer_vi, 3)),
          "vii - Miura EER=" + str(round(eer_vii, 3))],
         title="ROC")
