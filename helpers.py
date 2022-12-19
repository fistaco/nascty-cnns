import itertools
import multiprocessing as mp
import os
import pickle

import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy
CCE = CategoricalCrossentropy()

from constants import INVERSE_SBOX, SBOX, HW
from data_processing import shuffle_data
from metrics import keyrank, accuracy, MetricType


def exec_sca(ann_model, x_atk, y_atk, ptexts, true_subkey, subkey_idx=2):
    """
    Executes a side-channel attack on the given traces using the given neural
    network and returns the key rank obtained with the attack.
    """
    # Obtain y_pred_probs for each trace and aggregate them for the key guess
    y_pred_probs = ann_model.predict(x_atk)
    subkey_logprobs = subkey_pred_logprobs(y_pred_probs, ptexts, subkey_idx)

    return keyrank(subkey_logprobs, true_subkey)


def kfold_mean_inc_kr(y_pred_probs, ptexts, y_true, true_subkey, k, key_idx=2,
                      parallelise=False, hw=False, return_krs=False):
    """
    Computes the mean incremental key rank on the given list of prediction
    probability arrays & the data set's metadata over `k` folds.
    """
    # Store the incremental key rank for each fold
    set_size = len(y_pred_probs)
    inc_krs = np.zeros(k, dtype=np.float32)
    mean_krs = np.zeros(set_size, dtype=np.float32)

    if parallelise:
        pool = mp.Pool(get_pool_size())

        # Compute key ranks for each trace amount in parallel
        shuffled = [shuffle_data(y_pred_probs, ptexts, y_true) for _ in range(k)]
        argss = [
            (i, shuffled[i][0], shuffled[i][1], key_idx, set_size, true_subkey,
            False, hw)
            for i in range(k)
        ]
        # Result = 2D array indexed with [fold, trace_idx]
        map_results = pool.starmap(compute_fold_keyranks, argss)
        for fold in range(k):
            fold_krs, s = map_results[fold], shuffled[fold]
            inc_krs[fold] = incremental_keyrank(fold_krs, set_size, s[0], s[2])
            mean_krs += fold_krs/k

        pool.close()
        pool.join()
    else:
        for fold in range(k):
            y_pred_probs, ptexts, y_true = \
                shuffle_data(y_pred_probs, ptexts, y_true)
            fold_krs = compute_fold_keyranks(
                fold, y_pred_probs, ptexts, key_idx, set_size, true_subkey,
                verbose=True, hw=hw
            )
            inc_krs[fold] = incremental_keyrank(
                fold_krs, set_size, y_pred_probs, y_true
            )
            mean_krs += fold_krs/k

    if return_krs:
        return (np.mean(inc_krs), mean_krs)

    return np.mean(inc_krs)


def kfold_mean_key_ranks(y_pred_probs, ptexts, true_subkey, k, key_idx=2,
                         experiment_name="", parallelise=False, verbose=True,
                         hw=False):
    """
    Calculates and returns the mean key ranks of the given list of prediction
    probability arrays, the data set's metadata, and the desired number of
    folds.
    """
    # For each fold, store the key rank for all attack set sizes
    set_size = len(y_pred_probs)
    fold_key_ranks = np.zeros((set_size, k), dtype=np.uint8)

    # For both the parallel and sequential methods, the core idea is to reuse
    # subsets of the predictions to simulate attacks over different folds
    if parallelise:
        pool = mp.Pool(get_pool_size())

        # Compute key ranks for each trace amount in parallel
        shuffled = [shuffle_data(y_pred_probs, ptexts) for i in range(k)]
        argss = [
            (
                i, shuffled[i][0], shuffled[i][1], key_idx, set_size,
                true_subkey, verbose, hw
            )
            for i in range(k)
        ]
        # Result = 2D array indexed with [fold, trace_idx]
        map_results = pool.starmap(compute_fold_keyranks, argss)
        for fold in range(k):
            fold_key_ranks[:, fold] = map_results[fold]
    else:
        for fold in range(k):
            y_pred_probs, ptexts = shuffle_data(y_pred_probs, ptexts)
            fold_key_ranks[:, fold] = compute_fold_keyranks(
                fold, y_pred_probs, ptexts, key_idx, set_size, true_subkey,
                verbose=verbose, hw=hw
            )

    if parallelise:
        pool.close()
        pool.join()

    # Build an array that contains the mean key rank for each trace amount
    mean_key_ranks = np.zeros(set_size, dtype=np.uint8)
    for i in range(set_size):
        mean_key_ranks[i] = round(np.mean(fold_key_ranks[i]))
    if experiment_name:
        filepath = f"results/{experiment_name}_test_set_mean_key_ranks.pickle"
        with open(filepath, "wb") as f:
            pickle.dump(mean_key_ranks, f)

    # Print key ranks for various attack set sizes
    atk_set_sizes = range(set_size + 1, 100)
    reached_keyrank_zero = False
    if verbose:
        for (n_traces, rank) in enumerate(mean_key_ranks):
            # Print milestones
            if n_traces % 250 == 0:
                print(f"Mean key rank at {n_traces + 1} attack traces: {rank}")
            if rank == 0 and not reached_keyrank_zero:
                print(f"Key rank 0 obtained after {n_traces + 1} traces")
                reached_keyrank_zero = True
    
    return mean_key_ranks


def compute_fold_keyranks(fold, y_pred_probs, atk_ptexts, subkey_idx,
                          atk_set_size, true_subkey, verbose=True, hw=False):
    """
    Computes the key ranks for each number of traces for a given fold of attack
    traces, which are assumed to already be shuffled if necessary.
    """
    if verbose:
        print(f"Obtaining key ranks for fold {fold}...")
    fold_key_ranks = np.zeros(atk_set_size, dtype=np.uint8)

    # Track the summed log probability of each subkey candidate
    subkey_logprobs = np.zeros(256)

    # Iterate over each list of 256 probabilities in y_pred_probs.
    # Each list corresponds to the predictions of 1 trace.
    for (i, pred_probs) in enumerate(y_pred_probs):
        pt = atk_ptexts[i][subkey_idx]

        # TODO: Test the following implementation before using it for HW
        for k in range(256):
            l = compute_label(pt, k)
            if hw:
                l = HW[l]

            pred_prob = pred_probs[l]
            # Avoid computing np.log(0), which returns -inf
            logprob = np.log(pred_prob) if pred_prob > 0 \
                else min(pred_probs)**2
            subkey_logprobs[k] += logprob
    
        # Note that index i stores the key rank obtained after (i+1) traces
        fold_key_ranks[i] = keyrank(subkey_logprobs, true_subkey)

    return fold_key_ranks


def compute_fitness(nn, x_atk, y_atk, ptexts, metric_type, true_subkey,
                    atk_set_size, subkey_idx=2, hw=False, preds=None,
                    n_folds=1):
    """
    Executes a side-channel attack on the given traces using the given neural
    network and uses the obtained prediction probabilities to compute the key
    rank and/or accuracy.

    The `preds` argument allows precomputed predictions to be provided.
    """
    y_pred_probs = preds if preds is not None else nn.predict(x_atk)
    return evaluate_preds(
        y_pred_probs, metric_type, ptexts, true_subkey, y_atk, atk_set_size,
        subkey_idx, hw, n_folds
    )


def evaluate_preds(preds, metric_type, ptexts, k_true, true_labels, set_size,
                   k_idx=2, hw=False, n_folds=1):
    """
    Evaluates the given predictions using the method indicated by the given
    metric type and returns the result.
    """
    if metric_type == MetricType.KEYRANK:
        # subkey_probs = subkey_pred_logprobs(preds, ptexts, subkey_idx, hw)
        # return keyrank(subkey_probs, true_subkey)
        return kfold_mean_key_ranks(
                preds, ptexts, k_true, n_folds, k_idx, verbose=False, hw=hw
            )[-1]
    elif metric_type == MetricType.ACCURACY:
        return (1 - accuracy(preds, true_labels))*100
    elif metric_type == MetricType.KEYRANK_AND_ACCURACY:
        # subkey_probs = subkey_pred_logprobs(preds, ptexts, k_idx, hw)
        # res = keyrank(subkey_probs, k_true) - accuracy(preds, true_labels)
        kr = kfold_mean_key_ranks(
            preds, ptexts, k_true, n_folds, k_idx, verbose=False, hw=hw)[-1]
        return kr - accuracy(preds, true_labels)
    elif metric_type == MetricType.CATEGORICAL_CROSS_ENTROPY:
        return CCE(true_labels, preds).numpy()  # true_labels should be 1-hot
    elif metric_type == MetricType.INCREMENTAL_KEYRANK:
        # f = (n_traces_for_kr_zero + kr_10_pct + 0.5*kr_50_pct + 0.5*acc')
        if n_folds > 1:
            return multifold_inc_kr(
                preds, true_labels, ptexts, k_true, n_folds, k_idx,
                set_size, hw=hw
            )

        key_ranks = compute_fold_keyranks(
            7, preds, ptexts, k_idx, set_size, k_true, False, hw)

        # Adjust for inconsistencies
        element_wise_remaining_max_replace(key_ranks)

        return incremental_keyrank(key_ranks, set_size, preds, true_labels)
    elif metric_type == MetricType.KEYRANK_PROGRESS:
        if n_folds == 1:
            key_ranks = compute_fold_keyranks(
                7, preds, ptexts, k_idx, set_size, k_true, False, hw)
        else:
            key_ranks = kfold_mean_key_ranks(
                preds, ptexts, k_true, n_folds, key_idx=k_idx,
                verbose=False, hw=hw
            )

        return keyrank_progress(key_ranks, set_size)
    else:
        print("Encountered invalid metric type. Quitting.")
        exit(1)


def compute_label(ptext, subkey, mask=0):
    """
    Computes and returns the label, which is equivalent to the SBOX output, for
    a trace with a given plaintext and the targeted subkey.
    """
    return SBOX[ptext ^ subkey ^ mask]


def label_to_subkey(ptext, label, mask=0):
    """
    Retrieves the predicted subkey based on a predicted label and the known
    plaintext corresponding to some trace.
    """
    return INVERSE_SBOX[label] ^ ptext ^ mask


def subkey_pred_logprobs(label_pred_probs, ptexts, subkey_idx=2, hw=False):
    """
    Computes the logarithm of the prediction probability of each subkey
    candidate. This is done by iterating over each trace's label prediction
    probabilities and retrieving the corresponding subkey prediction
    probabilities.
    """
    # Track the summed log probability of each subkey candidate
    subkey_logprobs = np.zeros(256)

    # Iterate over each list of 256 probabilities in label_pred_probs
    for (i, pred_probs) in enumerate(label_pred_probs):
        pt = ptexts[i][subkey_idx]

        # TODO: Test the following implementation before using it for HW
        for k in range(256):
            l = compute_label(pt, k)
            if hw:
                l = HW[l]

            pred_prob = pred_probs[l]
            # Avoid computing np.log(0), which returns -inf
            logprob = np.log(pred_prob) if pred_prob > 0 \
                else min(pred_probs)**2
            subkey_logprobs[k] += logprob

    return subkey_logprobs


def incremental_keyrank(key_ranks, set_size, preds, true_labels):
    """
    Computes the incremental key rank metric as the % of traces at which kr 0
    was achieved + kr at 10% traces + 0.5*(kr at 50% traces) + 0.5*(1 - acc).
    """
    kr0_n_traces = first_zero_value_idx(key_ranks)/(set_size - 1)
    kr_10pct = min(key_ranks[round(set_size*0.1) - 1], 128)/128
    kr_50pct = min(key_ranks[round(set_size*0.5) - 1], 128)/128
    acc = accuracy(preds, true_labels)

    return kr0_n_traces + kr_10pct + 0.5*kr_50pct + 0.5*(1 - acc)


def multifold_inc_kr(preds, true_labels, pt, k_true, n_folds, k_idx, set_size,
                     hw=False):
    """
    Computes the incremental key rank metric over `n_folds` folds of the given
    predictions.
    """
    mean_ranks = kfold_mean_key_ranks(
        preds, pt, k_true, n_folds, key_idx=k_idx, verbose=False, hw=hw)
    return incremental_keyrank(mean_ranks, set_size, preds, true_labels)


def keyrank_progress(key_ranks, set_size):
    krs_20pct_steps = np.empty(9, dtype=np.uint8)
    krs_20pct_steps[0] = 128
    krs_20pct_steps[1:] = np.clip(key_ranks[::(set_size//8) - 1][1:], None, 128)

    # Adjust for inconsistencies
    element_wise_remaining_max_replace(krs_20pct_steps)

    return np.mean(np.diff(krs_20pct_steps.astype(np.int16)))


def ga_stagnation(fits, curr_gen, n_gens, thresh):
    """
    Determines whether a GA's progress has stagnated by observing if the
    fitness progress over `n_gens` generations is within `thresh`.
    """
    return (fits[max(0, curr_gen - n_gens)] - fits[curr_gen]) < thresh


def gen_nascty_exp_name(pop_size, max_gens, hw, polynom_mutation_eta,
                        crossover_type, truncation_proportion, noise=0.0,
                        desync=0):
    """
    Generates an experiment name for a NASCTY CNNs GA run using the given
    arguments.
    """
    lm_str = "hw" if hw else "id"
    co_str = f"{crossover_type.name.lower()}_co"

    return f"nascty-ps{pop_size}-{max_gens}gens-{lm_str}-" + \
           f"eta{polynom_mutation_eta}-{co_str}-tp{truncation_proportion}-" + \
           f"noise{noise}-desync{desync}"


def calc_max_fitness(metric_type):
    """
    Returns the maximum fitness based on the given metric type.
    """
    mapping = {
        MetricType.KEYRANK: 255,
        MetricType.KEYRANK_AND_ACCURACY: 255,
        MetricType.ACCURACY: 100,
        MetricType.CATEGORICAL_CROSS_ENTROPY: np.inf,
        MetricType.INCREMENTAL_KEYRANK: 3,
        MetricType.KEYRANK_PROGRESS: np.inf
    }
    return mapping[metric_type]


def calc_min_fitness(metric_type):
    """
    Returns the maximum fitness based on the given metric type.
    """
    mapping = {
        MetricType.KEYRANK: 0,
        MetricType.KEYRANK_AND_ACCURACY: -1,
        MetricType.ACCURACY: 0,
        MetricType.CATEGORICAL_CROSS_ENTROPY: 0,
        MetricType.INCREMENTAL_KEYRANK: 0,
        MetricType.KEYRANK_PROGRESS: -777,
    }
    return mapping[metric_type]


def get_pool_size(pop_size=50):
    """
    Determines and returns the size of the multiprocessing pool based on the
    given population size.
    """
    return round(min(pop_size*2, mp.cpu_count()*0.75))


def first_idx_with_value(a, x):
    """
    Finds the index in array a where element x first appears. Returns the last
    index if the element is not found.
    """
    for (i, n) in enumerate(a):
        if n == x:
            return i
    return len(a) - 1


def first_zero_value_idx(a):
    """
    finds the index in array a where value 0 first appears.
    """
    return first_idx_with_value(a, 0)


def element_wise_remaining_max_replace(a):
    """
    Replaces each element of `a` with the maximum value of all numbers after
    that element in `a`.
    """
    max_n = 0
    for i in range(len(a))[::-1]:
        if a[i] > max_n:
            max_n = a[i]
        else:
            a[i] = max_n


def consecutive_int_groups(a, stepsize=1):
    """
    Returns a list containing arrays of consecutive numbers in `a`.

    This code is based on the snippet by user 'unutbu' at https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    """
    if len(a) == 0:
        return []  # For an empty array, np.split would return [[]]

    split_idxs = np.where(np.diff(a) != stepsize)[0] + 1
    return np.split(a, split_idxs)


def is_categorical(labels):
    """
    Returns whether or not the given `labels` are formatted categorically by
    checking whether the inner layer has either 9 of 256 elements.
    """
    return len(np.shape(labels)) > 1 and np.shape(labels)[-1] > 1


def reshape_to_2d_singleton_array(xs):
    """
    Reshapes `xs` to a singleton array if it's 1-dimensional.

    This method is meant to be used to present NN layer information in a
    uniform format.
    """
    assert len(xs.shape) <= 2

    if len(xs.shape) == 1:
        return np.reshape(xs, (len(xs), 1))

    return xs
