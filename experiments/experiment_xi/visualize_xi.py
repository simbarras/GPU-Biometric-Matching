from visualize import *
plt.style.use('ggplot')

labels = ["i - quadrant 0 - same", "ii - quadrant 1 - same", "iii - quadrant 2 - same", "iv - quadrant 4 - same",
          "v - combined - same", "vi - combined - different"]

show_histogram(["v", "vi"], labels)

cam = None
eer_i, tpr_i, fpr_i = get_eer_confusion("i", "vi", "distance", cam=cam)
eer_ii, tpr_ii, fpr_ii = get_eer_confusion("ii", "vi", "distance", cam=cam)
eer_iii, tpr_iii, fpr_iii = get_eer_confusion("iii", "vi", "distance", cam=cam)
eer_iv, tpr_iv, fpr_iv = get_eer_confusion("iv", "vi", "distance", cam=cam)
eer_v, tpr_v, fpr_v = get_eer_confusion("v", "vi", "distance", cam=cam)

print("EERs:")
print(eer_i, eer_ii, eer_iii, eer_iv, eer_v)
show_roc([tpr_i, tpr_ii, tpr_iii, tpr_iv, tpr_v],
         [fpr_i, fpr_ii, fpr_iii, fpr_iv, fpr_v],
         ["i -  1st q. EER=" + str(round(eer_i, 3)),
          "ii -  2nd q. EER=" + str(round(eer_ii, 3)),
          "iii - 3rd q. EER=" + str(round(eer_iii, 3)),
          "iv - 4th q. EER=" + str(round(eer_iv, 3)),
          "v -  combined EER=" + str(round(eer_v, 3))],
         title="ROC")