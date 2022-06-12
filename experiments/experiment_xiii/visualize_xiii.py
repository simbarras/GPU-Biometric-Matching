from visualize import *
plt.style.use('ggplot')

labels = ["i - shift to mass - same", "ii - shift to mass - different",
          "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]

show_histogram(["xix", "xx"], labels)

cam = 2
eer_i, tpr_i, fpr_i = get_eer_confusion("xix", "xx", "distance", cam=cam)

print("EERs:")
print(eer_i)
show_roc([tpr_i],
         [fpr_i],
         ["Shift to Mass EER=" + str(round(eer_i, 3))],
         title="ROC")