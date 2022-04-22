from visualize import *

colors = ["skyblue", "maroon", "orange", "green", "grey"]
labels = ["Same Finger", "Same Finger", "Different Finger", "Different Finger"]

show_histogram(["i", "ii", "iii", "iv"], colors, labels)
eer, tpr, fpr = get_eer_confusion("i", "iii", "similarity")
eer_, tpr_, fpr_ = get_eer_confusion("ii", "iv", "similarity")

show_roc([tpr, tpr_], [fpr, fpr_], ["Edge Mask, EER: " + str(round(eer, 3)), "Fingerfocus Mask, EER: " + str(round(eer_, 3))], )