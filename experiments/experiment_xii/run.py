from experiment_runner import *
from experiment_setup import *

os.chdir("../..")
set_quadrant(0)
setup_experiment("xii")
run_experiment("xii", True)

set_quadrant(1)
setup_experiment("xii")
run_experiment("xii", True)

set_quadrant(2)
setup_experiment("xii")
run_experiment("xii", True)

set_quadrant(3)
setup_experiment("xii")
run_experiment("xii", True)