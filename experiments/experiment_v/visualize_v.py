from visualize import *
plt.style.use('ggplot')

labels = ["i - id - same", "ii - leftmost - same", "iii - huang - same", "iv - huang + leftmost - same", "v - huang + fingertip - same", "vi - CoM - same", "vii - miura - same", "viii - id - different", "ix - miura - different"]

show_histogram(["i", "viii", "ix"], labels)

eer_i, tpr_i, fpr_i = get_eer_confusion("i", "viii", "distance")
eer_ii, tpr_ii, fpr_ii = get_eer_confusion("ii", "viii", "distance")
eer_iii, tpr_iii, fpr_iii = get_eer_confusion("iii", "viii", "distance")
eer_iv, tpr_iv, fpr_iv = get_eer_confusion("iv", "viii", "distance")
eer_v, tpr_v, fpr_v = get_eer_confusion("v", "viii", "distance")
eer_vi, tpr_vi, fpr_vi = get_eer_confusion("vi", "viii", "distance")
eer_vii, tpr_vii, fpr_vii = get_eer_confusion("vii", "viii", "distance")

eer_i_m, tpr_i_m, fpr_i_m = get_eer_confusion("i", "ix", "distance")
eer_ii_m, tpr_ii_m, fpr_ii_m = get_eer_confusion("ii", "ix", "distance")
eer_iii_m, tpr_iii_m, fpr_iii_m = get_eer_confusion("ii", "ix", "distance")
eer_iv_m, tpr_iv_m, fpr_iv_m = get_eer_confusion("iv", "ix", "distance")
eer_v_m, tpr_v_m, fpr_v_m = get_eer_confusion("v", "ix", "distance")
eer_vi_m, tpr_vi_m, fpr_vi_m = get_eer_confusion("vi", "ix", "distance")
eer_vii_m, tpr_vii_m, fpr_vii_m = get_eer_confusion("vii", "ix", "distance")

print("Different Finger")
print(eer_i, eer_ii, eer_iii, eer_iv, eer_v, eer_vi, eer_vii)

print("Different Finger Miura Matching")
print(eer_i_m, eer_ii_m, eer_iii_m, eer_iv_m, eer_v_m, eer_vi_m, eer_vii_m)

show_roc([tpr_i, tpr_ii, tpr_iii, tpr_iv, tpr_v, tpr_vi, tpr_vii, tpr_vii],
         [fpr_i, fpr_ii, fpr_iii, fpr_iv, fpr_v, fpr_vi, fpr_vii, fpr_vii],
         ["i -  id EER: " + str(round(eer_i, 3)),
          "ii -  Leftm. EER=" + str(round(eer_ii, 3)),
          "iii - Huang EER=" + str(round(eer_iii, 3)),
          "iv - H + L EER=" + str(round(eer_iv, 3)),
          "v -  H + F EER=" + str(round(eer_v, 3)),
          "vi -  CoM EER=" + str(round(eer_vi, 3)),
          "vii - Miura EER=" + str(round(eer_vii, 3))],
         title="ROC vs. Different Finger Identity")

show_roc([tpr_i_m, tpr_ii_m, tpr_iii_m, tpr_iv_m, tpr_v_m, tpr_vi_m, tpr_vii_m, tpr_vii_m],
         [fpr_i_m, fpr_ii_m, fpr_iii_m, fpr_iv_m, fpr_v_m, fpr_vi_m, fpr_vii_m, fpr_vii_m],
         ["i -  id EER=" + str(round(eer_i_m, 3)),
          "ii -  Leftm. EER=" + str(round(eer_ii_m, 3)),
          "iii - Huang EER=" + str(round(eer_iii_m, 3)),
          "iv - H + L EER=" + str(round(eer_i_m, 3)),
          "v -  H + R EER=" + str(round(eer_v_m, 3)),
          "vi -  CoM EER=" + str(round(eer_vi_m, 3)),
          "vii - Miura EER=" + str(round(eer_vii_m, 3))],
         title="ROC vs. Different Finger Miura Matching")