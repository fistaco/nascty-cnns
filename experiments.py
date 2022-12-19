import multiprocessing as mp
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_processing import (load_ascad_data, load_data, sample_traces)
from helpers import (compute_fold_keyranks, exec_sca, kfold_mean_inc_kr,
                     gen_nascty_exp_name, first_zero_value_idx)
from metrics import MetricType, keyrank
from models import (build_small_cnn_ascad, train)
from nascty_cnns import NasctyCnnsGeneticAlgorithm
from nascty_enums import CrossoverType
from plotting import (plot_gens_vs_fitness, plot_n_traces_vs_key_rank,
                      plot_var_vs_key_rank)
from result_processing import ResultCategory, filter_df

def nascty_cnns_experiment(
    run_idx=0, max_gens=100, pop_size=100, parallelise=False, hw=False,
    select_fun="tournament", t_size=3, polynom_mutation_eta=20,
    crossover_type=CrossoverType.ONEPOINT,
    metric_type=MetricType.CATEGORICAL_CROSS_ENTROPY,
    truncation_proportion=1.0, n_valid_folds=1, n_atk_folds=100, noise=0.0,
    desync=0):
    """
    Runs a NASCTY CNNs genetic algorithm experiment on the ASCAD data set
    and stores the results in a directory specific to this experiment.
    """
    subkey_idx = 2
    (x_train, y_train, pt_train, k_train, x_atk, y_atk, pt_atk, k_atk) = \
        load_data("ascad", hw=hw, noise_std=noise, desync=desync)

    # Obtain balanced training and validation sets
    n_classes = 9 if hw else 256
    x_train, y_train, pt_train, x_val, y_val, pt_val = sample_traces(
        35584, x_train, y_train, pt_train, n_classes, balanced=True,
        return_remainder=True
    )
    x_val, y_val, pt_val = sample_traces(
        3840, x_val, y_val, pt_val, n_classes, balanced=True
    )

    exp_name = gen_nascty_exp_name(
        pop_size, max_gens, hw, polynom_mutation_eta, crossover_type,
        truncation_proportion, noise=noise, desync=desync
    )

    nascty_ga = NasctyCnnsGeneticAlgorithm(
        max_gens, pop_size, parallelise, select_fun, t_size,
        polynom_mutation_eta, crossover_type, metric_type,
        truncation_proportion, n_valid_folds
    )

    shuffle = n_valid_folds > 1
    best_indiv = nascty_ga.run(
        x_train, y_train, pt_train, x_val, y_val, pt_val, k_train, subkey_idx,
        shuffle, balanced=True, hw=hw, static_seed=True
    )

    print("Commencing training of best network.")
    nn = train(best_indiv.phenotype(hw=hw), x_train, y_train)

    print("Commencing evaluation on attack set.")
    y_pred_probs = nn.predict(x_atk)
    (inc_kr, mean_krs) = kfold_mean_inc_kr(
        y_pred_probs, pt_atk, y_atk, k_atk, n_atk_folds, subkey_idx,
        parallelise=parallelise, hw=hw, return_krs=True
    )

    # Save results in the proper experiment directory if run_idx is specified
    if run_idx >= 0:
        dir_path = f"nascty_results/{exp_name}"
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        filepath = f"{dir_path}/run{run_idx}_results.pickle"

        (_, best_fitness_per_gen, top_ten) = nascty_ga.get_results()
        results = (best_indiv, best_fitness_per_gen, top_ten, mean_krs, inc_kr)

        with open(filepath, "wb") as f:
            pickle.dump(results, f)


def results_from_exp_names(exp_names, exp_labels, file_tag, nascty=True):
    """
    Generates a fitness progress plot and key rank progress plot for one or
    more NASCTY-CNNs experiments with the given parameters. This method assumes
    the result files are already present in their respective directories.
    """
    n_repeats = 5
    fit_progress_arrays = []
    mean_krss = []
    inc_krs = []

    # Track variables for mean inter-exp comparison
    best_mean_inc_kr = 3.0
    best_exp_name = ""
    best_n_to_kr_zero = 77777  # Some arbitrary high number

    for (i, exp_name) in enumerate(exp_names):
        best_inc_kr = 3.0
        best_fit_progress_arr = None

        mean_inc_kr = 0.0

        dir_path = f"res/{exp_name}"
        for j in range(n_repeats):
            # Load results
            filepath = f"{dir_path}/run{j}_results.pickle"
            results = None
            with open(filepath, "rb") as f:
                results = pickle.load(f)
            inc_kr = results[-1]
            mean_inc_kr += inc_kr/n_repeats
            best_fitness_per_gen = results[1]

            inc_krs.append(inc_kr)

            if nascty:
                n_to_kr_zero = first_zero_value_idx(results[-2])
                print(f"{exp_name} run {j}: KR 0 in {n_to_kr_zero} traces")
                if n_to_kr_zero < best_n_to_kr_zero:
                    best_n_to_kr_zero = n_to_kr_zero

            if inc_kr < best_inc_kr:
                best_inc_kr = inc_kr
                best_fit_progress_arr = best_fitness_per_gen

                if nascty:
                    best_mean_krs = results[-2]

        if mean_inc_kr < best_mean_inc_kr:
            best_mean_inc_kr = mean_inc_kr
            best_exp_name = exp_name

        print(f"Mean INC_KR of {exp_name} = {mean_inc_kr}")

        plot_gens_vs_fitness(exp_labels[i], best_fit_progress_arr)
        fit_progress_arrays.append(best_fit_progress_arr)
        if nascty:
            mean_krss.append(best_mean_krs)

    print(f"Best exp = {best_exp_name} with mean inc. kr {best_mean_inc_kr}")
    print(f"Obtained KR 0 in {first_zero_value_idx(best_mean_krs)} traces")
    print(f"Smallest num. traces to KR 0 = {best_n_to_kr_zero}")

    labels = np.repeat(exp_labels, n_repeats)
    plot_var_vs_key_rank(labels, inc_krs, box=True, var_name="Experiment")
    plot_gens_vs_fitness(file_tag, *fit_progress_arrays,
                         labels=exp_labels)
    if nascty:
        plot_n_traces_vs_key_rank(file_tag, *mean_krss, labels=exp_labels)


def best_results_from_exp_name(exp_name):
    """
    Obtains the best tuple of results over all runs for the given `exp_name`.
    """
    n_repeats = 5
    dir_path = f"res/{exp_name}"

    # Find the best run and load the corresponding results
    best_inc_kr = 3.0
    best_results = None
    for i in range(n_repeats):
        # Load results
        filepath = f"{dir_path}/run{i}_results.pickle"
        results = None
        with open(filepath, "rb") as f:
            results = pickle.load(f)
        inc_kr = results[-1]

        if inc_kr < best_inc_kr:
            best_inc_kr = inc_kr
            best_results = results

    return best_results


def single_fold_multiple_nns_eval(fold, indices, atk_ptexts, subkey_idx,
                                  atk_set_size, nn_y_pred_probss,
                                  true_subkey, n_samples):
    """
    Computes the key rank for each given list of individual NN predictions
    over a single fold of the given data set with the given indices.
    """
    ptexts = atk_ptexts[indices]

    avg_fold_key_ranks = np.zeros(len(nn_y_pred_probss), dtype=float)
    for (i, y_pred_probs) in enumerate(nn_y_pred_probss):
        run_idx = i*30 + fold
        pred_probs = y_pred_probs[indices]
        trace_amnt_key_ranks = compute_fold_keyranks(
            run_idx, pred_probs, ptexts, subkey_idx, atk_set_size, true_subkey
        )

        avg_fold_key_ranks[i] = trace_amnt_key_ranks[n_samples - 1]

    return avg_fold_key_ranks


def ga_grid_search_parameter_influence_eval(eval_fitness=False):
    # Load df and params for best average performance
    with open("res/static_gs_weight_evo_results_df.pickle", "rb") as f:
        df = pickle.load(f)
    with open("res/static_gs_weight_evo_best_exp_data.pickle", "rb") as f:
        (mp, mr, mpdr, sf, tp, cor, exp_idx, inc_kr) = pickle.load(f)
    params = (mp, mr, mpdr, tp, cor)

    boxplot_cats = {
        ResultCategory.MUTATION_POWER_DECAY_RATE,
        ResultCategory.TRUNCATION_PROPORTION, ResultCategory.CROSSOVER_RATE
    }
    eval_cat = ResultCategory.FITNESS if eval_fitness \
        else ResultCategory.INCREMENTAL_KEYRANK

    # For each variable, plot its influence on the final key rank
    result_categories = list(ResultCategory)
    for result_cat in result_categories[:len(params)]:
        use_boxplot = result_cat in boxplot_cats

        sub_df = filter_df(df, params, exempt_idx=result_cat.value)
        plot_var_vs_key_rank(
            sub_df[:, result_cat.value],
            sub_df[:, eval_cat.value],
            result_cat,
            box=use_boxplot,
            eval_fitness=eval_fitness
        )

def mean_nascty_nn_eval():
    """
    Evaluates the NNs from the best NASCTY runs on synchronised and level 50
    desynchronised fixed-key ASCASD traces over 10 runs, each of which uses
    a different seed for the pseudorandom properties inherent to the evaluation
    process.  
    """
    n_repeats = 5
    exp_names = [
        "nascty-ps50-75gens-id-eta20-onepoint_co-tp1.0-noise0.0-desync0",
        "nascty-ps50-50gens-id-eta20-onepoint_co-tp1.0-noise0.0-desync50"
    ]
    exp_labels = ["Sync.", "Desync50"]
    run_idxs = [2, 1]
    desync_levels = [0, 50]

    mean_krss = []

    seeds = np.random.randint(2**31, size=10)

    for exp_name, exp_label, run_idx, desync in zip(exp_names, exp_labels, run_idxs, desync_levels):
        with open(f"res/{exp_name}/run{run_idx}_results.pickle", "rb") as f:
            genome = pickle.load(f)[0]

        best_traces_to_kr_zero = 10000
        mean_traces_to_kr_zero = 0
        best_mean_krs = None

        for i in range(n_repeats):
            np.random.seed(seeds[i])
            tf.random.set_seed(seeds[i])

            subkey_idx = 2
            (x_train, y_train, _, _, x_atk, y_atk, pt_atk, k_atk) = \
                load_data("ascad", hw=False, desync=desync)

            print(f"Training NN for {exp_name} run {i}")
            nn = train(genome.phenotype(hw=False), x_train, y_train)

            print("Commencing evaluation on attack set.")
            y_pred_probs = nn.predict(x_atk)
            (_, mean_krs) = kfold_mean_inc_kr(
                y_pred_probs, pt_atk, y_atk, k_atk, 100, subkey_idx, False,
                parallelise=True, hw=False, return_krs=True
            )

            traces_to_kr_zero = first_zero_value_idx(mean_krs)
            mean_traces_to_kr_zero += traces_to_kr_zero/n_repeats
            if traces_to_kr_zero < best_traces_to_kr_zero:
                best_traces_to_kr_zero = traces_to_kr_zero
                best_mean_krs = mean_krs

        print(f"{exp_name} fewest traces to KR zero: {best_traces_to_kr_zero}")
        print(f"{exp_name} mean traces to KR zero: {mean_traces_to_kr_zero}")
        mean_krss.append(best_mean_krs)

    file_tag = "NASCTY_final_eval."
    plot_n_traces_vs_key_rank(file_tag, *mean_krss, labels=exp_labels)


def construct_nascty_dirs(argss, only_print=False):
    """
    Constructs a directory for the experiment names corresponding to the given
    NASCTY-CNNs argument lists.
    """
    for args in argss:
        exp_name = gen_nascty_exp_name(*args)
        dir_path = f"nascty_results/{exp_name}"

        if not os.path.exists(dir_path) and not only_print:
            os.mkdir(dir_path)

        print(f"Constructed {dir_path}")
