import multiprocessing as mp

mp.set_start_method("spawn", force=True)
import os
import sys

# The following parameters enable multiprocessing with tensorflow on a CPU
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

tf.get_logger().setLevel("ERROR")  # Hides warnings when using small data sets
from tensorflow import keras

from data_processing import load_ascad_data, train_test_split
from experiments import (nascty_cnns_experiment, results_from_exp_names,
                         construct_nascty_dirs)
from helpers import gen_nascty_exp_name
from metrics import MetricType
from nascty_enums import CrossoverType

# =============================================================================
nascty_configs = [
    # psize, max_gens, hw, polynom_mut_eta, co_type, trunc_prop, noise, desync
    (26, 10, False, 20, CrossoverType.ONEPOINT, 0.6, 0.0, 0),
    # Grid search parameters start here (@ configs[1])
    (26, 10, False, 20, CrossoverType.ONEPOINT, 0.5, 0.0, 0),
    (26, 10, False, 20, CrossoverType.ONEPOINT, 1.0, 0.0, 0),
    (26, 10, False, 20, CrossoverType.PARAMETERWISE, 0.5, 0.0, 0),
    (26, 10, False, 20, CrossoverType.PARAMETERWISE, 1.0, 0.0, 0),
    (26, 10, False, 40, CrossoverType.ONEPOINT, 0.5, 0.0, 0),
    (26, 10, False, 40, CrossoverType.ONEPOINT, 1.0, 0.0, 0),
    (26, 10, False, 40, CrossoverType.PARAMETERWISE, 0.5, 0.0, 0),
    (26, 10, False, 40, CrossoverType.PARAMETERWISE, 1.0, 0.0, 0),
    # Grid search parameters end here (@ configs[8])
    # ASCAD full resource experiments, masked (configs[9] and configs[10])
    (26, 50, False, 20, CrossoverType.ONEPOINT, 1.0, 0.0, 0),  # psize 52, 50 gens
    (50, 75, False, 20, CrossoverType.ONEPOINT, 1.0, 0.0, 0),  # psize 100, 75 gens
    # Desync experiments start here (configs[11] - configs[13]) (psize 100, 50 gens)
    (50, 50, False, 20, CrossoverType.ONEPOINT, 1.0, 0.0, 10),
    (50, 50, False, 20, CrossoverType.ONEPOINT, 1.0, 0.0, 30),
    (50, 50, False, 20, CrossoverType.ONEPOINT, 1.0, 0.0, 50)
]
cf = nascty_configs[int(sys.argv[1])]

if __name__ == "__main__":
    # Run NASCTY with hard-coded variables
    nascty_cnns_experiment(
        run_idx=-1, max_gens=3, pop_size=4, parallelise=True, hw=False, select_fun="tournament", t_size=3, polynom_mutation_eta=20, crossover_type=CrossoverType.ONEPOINT,
        metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY, truncation_proportion=0.5, n_valid_folds=1, n_atk_folds=5, noise=0.0, desync=0
    )

    # Run NASCTY with variables taken from a config tuple named "cf"
    # nascty_cnns_experiment(
    #     run_idx=int(sys.argv[2]), max_gens=cf[1], pop_size=cf[0], parallelise=True, remote=True, hw=cf[2], select_fun="tournament", t_size=3, polynom_mutation_eta=cf[3], crossover_type=cf[4],
    #     metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY, truncation_proportion=cf[5], n_valid_folds=1, n_atk_folds=100, noise=cf[6], desync=cf[7]
    # )

    # Construct the directories in which NASCTY results will be stored
    # construct_nascty_dirs(nascty_configs)

    # Example with which result plots can be generated
    # exp_names = [gen_nascty_exp_name(*args) for args in nascty_configs[10:]]
    # exp_labels = ["Sync", "Desync10", "Desync30", "Desync50"]
    # file_tag = "NASCTY_full_exps"
    # results_from_exp_names(exp_names, exp_labels, file_tag, nascty=True)
