from pathlib import *
import sys
import os
# sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import time
import argparse

from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, getp, axes, subplot

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def process_timing_data_from_file(path_to_data):

    timings = []
    with open(path_to_data, "r") as file:
        for line in file:
            parts = line.split(",")

            for time in parts[1:]:
                timings.append(float(time))

    return timings

def setBoxColors(bp):
    if len(bp['boxes']) > 0:
        setp(bp['boxes'][0], color='blue')
        setp(bp['caps'][0], color='blue')
        setp(bp['caps'][1], color='blue')
        setp(bp['whiskers'][0], color='blue')
        setp(bp['whiskers'][1], color='blue')
        #setp(bp['fliers'][0], color='blue')
        setp(bp['medians'][0], color='blue')
    if len(bp['boxes']) > 1:
        setp(bp['boxes'][1], color='red')
        setp(bp['caps'][2], color='red')
        setp(bp['caps'][3], color='red')
        setp(bp['whiskers'][2], color='red')
        setp(bp['whiskers'][3], color='red')
        #setp(bp['fliers'][1], color='red')
        setp(bp['medians'][1], color='red')

    if len(bp['boxes']) > 2:
        setp(bp['boxes'][2], color='green')
        setp(bp['caps'][4], color='green')
        setp(bp['caps'][5], color='green')
        setp(bp['whiskers'][4], color='green')
        setp(bp['whiskers'][5], color='green')
        #setp(bp['fliers'][1], color='green')
        setp(bp['medians'][2], color='green')

# cases per bit {1: total_pip (000001), 2: pipelineSteps (000010), 3: matchingSteps (000100), 
#                4: data_from_py (001000), 5: data_from_cpp (010000), 6: data_from_cpp_opt (100000)}

parser = argparse.ArgumentParser(description='Program to process timing data and generate a gnuplot graph.')

parser.add_argument('-sp-c-edm', '--pipeline-steps-path-cpp-edge-mask', dest='steps_pipeline_path_cpp_edge', type=Path, help='The path leading to the data corresponding to the time measurements of the edge mask extraction execution on C++.')
parser.add_argument('-sp-p-edm', '--pipeline-steps-path-py-edge-mask', dest='steps_pipeline_path_py_edge', type=Path, help='The path leading to the data corresponding to the time measurements of the edge mask extraction execution on Python.')
parser.add_argument('-sp-co-edm', '--pipeline-steps-path-cpp-opt-edge-mask', dest='steps_pipeline_path_cpp_opt_edge', type=Path, help='The path leading to the data corresponding to the time measurements of the optimized edge mask extraction execution on C++.')

parser.add_argument('-sp-c-pr', '--pipeline-steps-path-cpp-prealignment', dest='steps_pipeline_path_cpp_pre', type=Path, help='The path leading to the data corresponding to the time measurements of the prealignment execution on C++.')
parser.add_argument('-sp-p-pr', '--pipeline-steps-path-py-prealignment', dest='steps_pipeline_path_py_pre', type=Path, help='The path leading to the data corresponding to the time measurements of the prealignment execution on Python.')
parser.add_argument('-sp-co-pr', '--pipeline-steps-path-cpp-opt-prealignment', dest='steps_pipeline_path_cpp_opt_pre', type=Path, help='The path leading to the data corresponding to the time measurements of the optimized prealignment execution on C++.')

parser.add_argument('-mp-c-po', '--pipeline-match-path-cpp-postalignment', dest='match_pipeline_path_cpp_post', type=Path, help='The path leading to the data corresponding to the time measurements of the postalignment execution on C++.')
parser.add_argument('-mp-p-po', '--pipeline-match-path-py-postalignment', dest='match_pipeline_path_py_post', type=Path, help='The path leading to the data corresponding to the time measurements of the postalignment execution on Python.')
parser.add_argument('-mp-co-po', '--pipeline-match-path-cpp-opt-postalignment', dest='match_pipeline_path_cpp_opt_post', type=Path, help='The path leading to the data corresponding to the time measurements of the optimized postalignment execution on C++.')

parser.add_argument('-mp-c-mm', '--pipeline-match-path-cpp-matching', dest='match_pipeline_path_cpp_mm', type=Path, help='The path leading to the data corresponding to the time measurements of the miura matching execution on C++.')
parser.add_argument('-mp-p-mm', '--pipeline-match-path-py-matching', dest='match_pipeline_path_py_mm', type=Path, help='The path leading to the data corresponding to the time measurements of the miura matching execution on Python.')
parser.add_argument('-mp-co-mm', '--pipeline-match-path-cpp-opt-matching', dest='match_pipeline_path_cpp_opt_mm', type=Path, help='The path leading to the data corresponding to the time measurements of the optimized miura matching execution on C++.')


args = parser.parse_args()

edgeM = []
preal = []
postal = []
miuMat = []
labels = []

# Add data for each input of edge mask timing to group edgeM
if args.steps_pipeline_path_py_edge is not None:
    edgeM.append(process_timing_data_from_file(args.steps_pipeline_path_py_edge))
else:
    edgeM.append([])

if args.steps_pipeline_path_cpp_edge is not None:
    edgeM.append(process_timing_data_from_file(args.steps_pipeline_path_cpp_edge))
else:
    edgeM.append([])

if args.steps_pipeline_path_cpp_opt_edge is not None:
    edgeM.append(process_timing_data_from_file(args.steps_pipeline_path_cpp_opt_edge))
else:
    edgeM.append([])

if len(edgeM) != 0:
    labels.append('Edge Mask Extraction')

# Add data for each input of prealignment timing to group preal
if args.steps_pipeline_path_py_pre is not None:
    preal.append(process_timing_data_from_file(args.steps_pipeline_path_py_pre))
else:
    preal.append([])

if args.steps_pipeline_path_cpp_pre is not None:
    preal.append(process_timing_data_from_file(args.steps_pipeline_path_cpp_pre))
else:
    preal.append([])

if args.steps_pipeline_path_cpp_opt_pre is not None:
    preal.append(process_timing_data_from_file(args.steps_pipeline_path_cpp_opt_pre))
else:
    preal.append([])

if len(preal) != 0:
    labels.append('Prealignment')

# Add data for each input of postalignment timing to group postal
if args.match_pipeline_path_py_post is not None:
    postal.append(process_timing_data_from_file(args.match_pipeline_path_py_post))
else:
    postal.append([])

if args.match_pipeline_path_cpp_post is not None:
    postal.append(process_timing_data_from_file(args.match_pipeline_path_cpp_post))
else:
    postal.append([])

if args.match_pipeline_path_cpp_opt_post is not None:
    postal.append(process_timing_data_from_file(args.match_pipeline_path_cpp_opt_post))
else:
    postal.append([])

if len(postal) != 0:
    labels.append('Postalignment')

# Add data for each input of miura matching timing to group miuMat
if args.match_pipeline_path_py_mm is not None:
    miuMat.append(process_timing_data_from_file(args.match_pipeline_path_py_mm))
else:
    miuMat.append([])

if args.match_pipeline_path_cpp_mm is not None:
    miuMat.append(process_timing_data_from_file(args.match_pipeline_path_cpp_mm))
else:
    miuMat.append([])

if args.match_pipeline_path_cpp_opt_mm is not None:
    miuMat.append(process_timing_data_from_file(args.match_pipeline_path_cpp_opt_mm))
else:
    miuMat.append([])

if len(miuMat) != 0:
    labels.append('Distance Matching')

labels.append('')

edgeM_size = len(edgeM)
preal_size = len(preal)
postal_size = len(postal)
miuMat_size = len(miuMat)

# Start boxplot generation
fig= figure()
ax = axes()

ticks = []

# TODO: more if statements to ensure correct generation even without data


pos2 = list(range(1, 1 + edgeM_size))
sizePos = edgeM_size
if edgeM_size > 0:
    ticks.append(((pos2[-1] - pos2[0]) / 2) + pos2[0])
pos3 = list(range(sizePos + 2, sizePos + 2 + preal_size))
sizePos += preal_size + 1
if preal_size > 0:
    ticks.append(((pos3[-1] - pos3[0]) / 2) + pos3[0])
pos5 = list(range(sizePos + 2, sizePos + 2 + postal_size))
sizePos += postal_size + 1
if postal_size > 0:
    ticks.append(((pos5[-1] - pos5[0]) / 2) + pos5[0])
pos6 = list(range(sizePos + 2, sizePos + 2 + miuMat_size))
if miuMat_size > 0:
    ticks.append(((pos6[-1] - pos6[0]) / 2) + pos6[0])

ticks.append(ticks[-1] + 2)


if edgeM_size > 0:
    bp = boxplot(edgeM, sym = '', positions = pos2, widths = 0.6)
    setBoxColors(bp)

if preal_size > 0:
    bp = boxplot(preal, sym = '', positions = pos3, widths = 0.6)
    setBoxColors(bp)

if postal_size > 0:
    bp = boxplot(postal, sym = '', positions = pos5, widths = 0.6)
    setBoxColors(bp)

if miuMat_size > 0:
    bp = boxplot(miuMat, sym = '', positions = pos6, widths = 0.6)
    setBoxColors(bp)

xlim(0,5)
ylim(0,0.1)
#labels = ['Total', 'Edge', 'Pre', 'MaxC', 'Post', 'Dist', '']

ax.set_xticks(ticks)
ax.set_xticklabels(labels)
#ax.set_xlabel('Help', color='k', labelpad=15)


hB, = plot([1,1],'b-')
hR, = plot([1,1],'r-')
hG, = plot([1,1],'g-')
legend((hB, hR, hG),('C++', 'Python', 'C++ Optimized'))
hB.set_visible(False)
hR.set_visible(False)
hG.set_visible(False)

use_sticky_edges = False

savefig('boxcompare_steps.png')