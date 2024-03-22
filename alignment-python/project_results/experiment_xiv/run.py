from experiment_runner import *
from experiment_setup import *

os.chdir("../..")

for i in range(4):
    for j in range(4):
        set_quadrant([i, j])
        setup_experiment("xiv", 1592)
        run_experiment("xiv", True)