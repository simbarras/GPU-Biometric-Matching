from resources import *
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
from experiment_setup import tuple_to_filename
from experiment_setup import num_to_roman
import json
from os.path import isfile, join

######################## globals
experiment_dir_pref = "experiments/experiment_"
dataset_dir_pref = "dataset_"
experiment_id = "i"

def calculate_distances(idx, dataset_path, setup_in_path, results_out_path):
    df = pd.read_csv(setup_in_path)
    for row in df.itertuples():
        row = list(row)
        i = row[0]
        tpl = row[1:]

        ###################################################################### Useful Paths declaration
        print(tuple_to_filename(tpl, idx, ".png"))
        model_path_png, probe_path_png = tuple_to_filename(tpl, idx, ".png")
        model_path_png, probe_path_png = dataset_path + model_path_png, dataset_path + probe_path_png

        ###################################################################### Run extraction pipeline
        model = run_pipeline(model_path_png, caching=0,
                             mask_method=tpl[idx["mask"]],
                             prealign_method=tpl[idx["prealign"]],
                             preprocess_method=tpl[idx["preprocess"]],
                             extraction_method=tpl[idx["feature_extractor"]],
                             postprocess_method=tpl[idx["postprocess"]],
                             postalign_method=tpl[idx["postalign"]])

        probe = run_pipeline(probe_path_png, caching=0,
                             mask_method=tpl[idx["mask"]],
                             prealign_method=tpl[idx["prealign"]],
                             preprocess_method=tpl[idx["preprocess"]],
                             extraction_method=tpl[idx["feature_extractor"]],
                             postprocess_method=tpl[idx["postprocess"]],
                             postalign_method=tpl[idx["postalign"]],
                             model=model)

        ###################################################################### Compute distance
        distance = compute_single_distance(model, probe, tpl[idx["distance_function"]])

        ###################################################################### Update dataframe
        df.at[i, "distance"] = distance
    df.to_csv(results_out_path)


# assumes experiment folder with initialized specs already exists.
def run_population_experiment(experiment_id='i', population_id='i'):
    experiment_path = experiment_dir_pref + experiment_id + "/population_" + population_id

    # load specification and idx
    f = open(experiment_path + "/spec.json")
    experiment_spec = json.load(f)
    f.close()

    # calculate distances of dataframe
    dataset_path = dataset_dir_pref + experiment_spec["spec"]["dataset_id"][0] + "/"
    calculate_distances(idx=experiment_spec["idx"],
                        dataset_path=dataset_path,
                        setup_in_path=experiment_path + "/setup.csv",
                        results_out_path=experiment_path + "/results.csv")


def run_experiment(experiment_id='i'):
    """
    Runs given experiment. Only computes result if no results.csv file exists for given population.
    """
    experiment_path = "experiments/experiment_" + experiment_id

    # go through all valid populations
    i = 1
    while(True):
        p = num_to_roman(i)
        population_path = experiment_path + "/population_" + p
        if os.path.isdir(population_path):
            print("Loading population " + p + "...")
            if not os.path.isfile(population_path + "results.csv"):
                run_population_experiment(experiment_id, p)
        else:
            break
        i = i + 1
    print("FINISHED")

if __name__ == "__main__":
    run_experiment(experiment_id)

