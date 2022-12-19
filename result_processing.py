from params import SELECTION_FUNCTION
import pickle
from enum import Enum
import sys

import numpy as np


def filter_df(df, variables, exempt_idx=-1):
    """
    Returns a filtered data frame (DF) containing the DF rows of which the
    values in the columns correspond to the given variables, with the
    exemption of the variable of a given exempt index.
    """
    df_sub = df
    for (i, var) in enumerate(variables):
        if i == exempt_idx:
            continue
        df_sub = df_sub[df_sub[:, i] == variables[i]]

    return df_sub


def load_fitness_vs_keyrank_results_df(exp_name, n_experiments=10):
    """
    Constructs a dataframe containing the key rank and fitness value form GA
    run results of multiple experiments with parameters according to the given
    experiment name. 
    """
    df = np.zeros((n_experiments, 2), dtype=np.float32)
    dir_path = f"results/{exp_name}"

    for i in range(n_experiments):
        with open(f"{dir_path}/run{i}_results.pickle", "rb") as f:
            (best_indiv, _, _, key_rank) = pickle.load(f)
        df[i] = [best_indiv.fitness, key_rank]

    return df


def fitness_keyrank_corr(df, fit_col_idx=0, kr_col_idx=1):
    """
    Computes and returns the correlation between the fitness and key rank
    columns in the given DataFrame (DF). If no column indices are given, the
    DF should solely have a fitness column and a keyrank column. 
    """
    return np.corrcoef(df[:, [fit_col_idx, kr_col_idx]].T)[0, 1]


class ResultCategory(Enum):
    MUTATION_POWER = 0
    MUTATION_RATE = 1
    MUTATION_POWER_DECAY_RATE = 2
    TRUNCATION_PROPORTION = 3
    CROSSOVER_RATE = 4
    FITNESS = 5
    INCREMENTAL_KEYRANK = 6
    SELECTION_FUNCTION = 7
