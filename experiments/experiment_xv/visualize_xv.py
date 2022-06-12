from visualize import *
plt.style.use('ggplot')

labels = ["i - shift to mass - same", "ii - shift to mass - different"]

show_histogram(["i", "ii"], labels)

cam = None
eer_i, tpr_i, fpr_i = get_eer_confusion("i", "ii", "distance", cam=cam)

print("EERs:")
print(eer_i)
show_roc([tpr_i],
         [fpr_i],
         ["i -  Ident. EER=" + str(round(eer_i, 3))],
         title="ROC")