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
parser.add_argument('-pp-c', '--pipeline-path-cpp', dest='total_pipeline_path_cpp', type=Path, help='The path leading to the data corresponding to the time measurements of the total pipeline execution on C++.')
parser.add_argument('-pp-p', '--pipeline-path-py', dest='total_pipeline_path_py', type=Path, help='The path leading to the data corresponding to the time measurements of the total pipeline execution on Python.')
parser.add_argument('-pp-co', '--pipeline-path-cpp-opt', dest='total_pipeline_path_cpp_opt', type=Path, help='The path leading to the data corresponding to the time measurements of the optimized total pipeline execution on C++.')

parser.add_argument('-sp-c-mc', '--pipeline-steps-path-cpp-maxCurv', dest='steps_pipeline_path_cpp_maxc', type=Path, help='The path leading to the data corresponding to the time measurements of the maximum curvature execution on C++.')
parser.add_argument('-sp-p-mc', '--pipeline-steps-path-py-maxCurv', dest='steps_pipeline_path_py_maxc', type=Path, help='The path leading to the data corresponding to the time measurements of the maximum curvature execution on Python.')
parser.add_argument('-sp-co-mc', '--pipeline-steps-path-cpp-opt-maxCurv', dest='steps_pipeline_path_cpp_opt_maxc', type=Path, help='The path leading to the data corresponding to the time measurements of the optimized maximum curvature execution on C++.')

args = parser.parse_args()

total = []
maxCurv = []
labels = []

# Add data for each input of total pipeline timing to group total
if args.total_pipeline_path_py is not None:
    total.append(process_timing_data_from_file(args.total_pipeline_path_py))
else:
    total.append([])

if args.total_pipeline_path_cpp is not None:
    total.append(process_timing_data_from_file(args.total_pipeline_path_cpp))
else:
    total.append([])

if args.total_pipeline_path_cpp_opt is not None:
    total.append(process_timing_data_from_file(args.total_pipeline_path_cpp_opt))
else:
    total.append([])

if len(total) != 0:
    labels.append('Total Pipeline')

# Add data for each input of maximum curvature timing to group maxCurv
if args.steps_pipeline_path_py_maxc is not None:
    maxCurv.append(process_timing_data_from_file(args.steps_pipeline_path_py_maxc))
else:
    maxCurv.append([])

if args.steps_pipeline_path_cpp_maxc is not None:
    maxCurv.append(process_timing_data_from_file(args.steps_pipeline_path_cpp_maxc))
else:
    maxCurv.append([])

if args.steps_pipeline_path_cpp_opt_maxc is not None:
    maxCurv.append(process_timing_data_from_file(args.steps_pipeline_path_cpp_opt_maxc))
else:
    maxCurv.append([])

if len(maxCurv) != 0:
    labels.append('Maximum Curvature')

labels.append('')

total_size = len(total)
maxCurv_size = len(maxCurv)

# Start boxplot generation
fig= figure()
ax = axes()

ticks = []

# TODO: more if statements to ensure correct generation even without data

pos1 = list(range(1, total_size + 1))
sizePos = total_size
if total_size > 0:
    ticks.append(((pos1[-1] - pos1[0]) / 2) + pos1[0])
pos2 = list(range(sizePos + 2, sizePos + 2 + maxCurv_size))
if maxCurv_size > 0:
    ticks.append(((pos2[-1] - pos2[0]) / 2) + pos2[0])

ticks.append(ticks[-1] + 2)


if total_size > 0:
    bp = boxplot(total, sym = '', positions = pos1, widths = 0.6)
    setBoxColors(bp)
if maxCurv_size > 0:
    bp = boxplot(maxCurv, sym = '', positions = pos2, widths = 0.6)
    setBoxColors(bp)

xlim(0,5)
ylim(0,2.3)
#set y ticks to more

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

savefig('boxcompare_tot.png')