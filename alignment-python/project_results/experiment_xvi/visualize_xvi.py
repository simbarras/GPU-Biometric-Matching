from visualize import *
plt.style.use('ggplot')

labels = ["i - opt miura - same", "ii - opt miura - different", "iii - opt trans - same", "iv - opt trans - different"]

show_histogram(["i", "ii"], labels)

cam = None
eer_i, tpr_i, fpr_i = get_eer_confusion("i", "ii", "distance", cam=cam)
eer_ii, tpr_ii, fpr_ii = get_eer_confusion("iii", "iv", "distance", cam=cam)
print(eer_i, len(tpr_i), len(tpr_ii))
print("EERs:")
print(eer_i, eer_ii)
show_roc([tpr_i, tpr_ii],
         [fpr_i, fpr_i],
         ["i -  Ident. EER=" + str(round(eer_i, 3)),
          "ii -  Leftm. EER=" + str(round(eer_ii, 3))],
         title="ROC")
