from visualize import *
plt.style.use('ggplot')

labels = ["i - shift to mass - same", "ii - shift to mass - different",
          "", "", "", "", "", "", "", "", "", "", "", "", "", "", "Mass Alignment - 16 retries- Same", "Mass Alignment - Different", "", "", "", ""]

show_histogram(["xvii", "x",  "xviii"], labels)

cam = None
eer_i, tpr_i, fpr_i = get_eer_confusion("i", "xix", "distance", cam=cam)
eer_ii, tpr_ii, fpr_ii = get_eer_confusion("ii", "xx", "distance", cam=cam)
eer_iii, tpr_iii, fpr_iii = get_eer_confusion("iii", "xxi", "distance", cam=cam)
eer_iv, tpr_iv, fpr_iv = get_eer_confusion("iv", "xxii", "distance", cam=cam)
eer_v, tpr_v, fpr_v = get_eer_confusion("v", "xxiii", "distance", cam=cam)
eer_vi, tpr_vi, fpr_vi = get_eer_confusion("vi", "xxiv", "distance", cam=cam)
eer_vii, tpr_vii, fpr_vii = get_eer_confusion("vii", "xxv", "distance", cam=cam)
eer_viii, tpr_viii, fpr_viii = get_eer_confusion("viii", "xxvi", "distance", cam=cam)
eer_ix, tpr_ix, fpr_ix = get_eer_confusion("ix", "xxvii", "distance", cam=cam)
eer_x, tpr_x, fpr_x = get_eer_confusion("x", "xxviii", "distance", cam=cam)
eer_xi, tpr_xi, fpr_xi = get_eer_confusion("xi", "xxix", "distance", cam=cam)
eer_xii, tpr_xii, fpr_xii = get_eer_confusion("xii", "xxx", "distance", cam=cam)
eer_xiii, tpr_xiii, fpr_xiii = get_eer_confusion("xiii", "xxxi", "distance", cam=cam)
eer_xiv, tpr_xiv, fpr_xiv = get_eer_confusion("xiv", "xxxii", "distance", cam=cam)
eer_xv, tpr_xv, fpr_xv = get_eer_confusion("xv", "xxxiii", "distance", cam=cam)
eer_xvi, tpr_xvi, fpr_xvi = get_eer_confusion("xvi", "xxxiv", "distance", cam=cam)
eer_xvii, tpr_xvii, fpr_xvii = get_eer_confusion("xvii", "xviii", "distance", cam=cam)

print("EERs:")
print(eer_i)
show_roc([tpr_i, tpr_ii, tpr_iii, tpr_iv, tpr_v, tpr_vi, tpr_vii, tpr_viii, tpr_ix, tpr_x, tpr_xi, tpr_xii, tpr_xiii, tpr_xiv, tpr_xv, tpr_xvi, tpr_xvii],
         [fpr_i, fpr_ii, fpr_iii, fpr_iv, fpr_v, fpr_vi, fpr_vii, fpr_viii, fpr_ix, fpr_x, fpr_xi, fpr_xii, fpr_xiii, fpr_xiv, fpr_xv, fpr_xvi, fpr_xvii],
         ["[0, 0] EER=" + str(round(eer_i, 3)),
          "[0, 1] EER=" + str(round(eer_ii, 3)),
          "[0, 2] EER=" + str(round(eer_iii, 3)),
          "[0, 3] EER=" + str(round(eer_iv, 3)),
          "[1, 0] EER=" + str(round(eer_v, 3)),
          "[1, 1] EER=" + str(round(eer_vi, 3)),
          "[1, 2] EER=" + str(round(eer_vii, 3)),
          "[1, 3] EER=" + str(round(eer_viii, 3)),
          "[2, 0] EER=" + str(round(eer_ix, 3)),
          "[2, 1] EER=" + str(round(eer_x, 3)),
          "[2, 2] EER=" + str(round(eer_xi, 3)),
          "[2, 3] EER=" + str(round(eer_xii, 3)),
          "[3, 0] EER=" + str(round(eer_xiii, 3)),
          "[3, 1] EER=" + str(round(eer_xiv, 3)),
          "[3, 2] EER=" + str(round(eer_xv, 3)),
          "[3, 3] EER=" + str(round(eer_xvi, 3)),
          "Combined EER=" + str(round(eer_xvii, 3)),],
         title="ROC")