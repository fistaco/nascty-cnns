This repository contains the source code for _Neuroevolution to Attack Side-Channel Traces Yielding Convolutional Neural Networks_ (NASCTY-CNNs), a genetic algorithm that evolves the parameters of CNNs for side-channel analysis.

The contents of the ```.py``` files are self-explanatory and relevant Keras model files can be found in the ```trained_networks``` directory. Furthermore, note that this repository does not contain the ASCAD and ChipWhisperer data sets, so you will have to provide these yourself and assign their paths to the `ASCAD_DIRECTORY` or `CHIPWHISPERER_DIRECTORY` variables in `data_processing.py`.

For knowledge on NASCTY's inner workings, refer to the `NasctyCnnsGeneticAlgorithm.run(...)` method in `nascty_cnns.py` and follow the method calls from its main loop.

# Usage
NASCTY can be run from `main.py` in two ways: you can either hard-code your own parameters in the method call to `nascty_cnns_experiment` or you can define a list of tuples in which each tuple defines a NASCTY parameter configuration. You can then run `python3 main.py n` where `n` represents the n-th configuration tuple to run NASCTY. An example of this is provided below.
```
nascty_configs = [
    # psize, max_gens, hw, polynom_mut_eta, co_type, trunc_prop, noise, desync
    (26, 10, False, 20, CrossoverType.PARAMETERWISE, 0.5, 0.0, 0),
    (26, 50, False, 20, CrossoverType.ONEPOINT, 1.0, 0.0, 0),  # psize 52, 50 gens
    (50, 75, False, 20, CrossoverType.ONEPOINT, 1.0, 0.0, 0),  # psize 100, 75 gens
]
cf = nascty_configs[int(sys.argv[1])]
```
