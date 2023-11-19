# Training robust and generalizable quantum models
This repository contains the source code, data and plots for the paper 
"Training robust and generalizable quantum models" 
by Julian Berberich, Daniel Fink, Daniel Pranjić, Christian Tutschku and Christian Holm [[arXiv:TBA](https://arxiv.org/abs/TBA)].

## Installation
This repository uses [poetry](https://python-poetry.org) for dependency management.
In order to run the numerical simulations, poetry must be [installed first](https://python-poetry.org/docs/#installation).
Afterwards, run 

``` bash
poetry install
```

from the root directory of the repository to setup the Python virtual environment.

## Overview

The repository is contains the following directories:

- __circle1__: contains all scripts for creating and visualizing the numerical simulations.
- __data__: directory for storing the numerical results as raw data.
- __plots__: contains all the plots as PDFs.

## Running Numerical Simulations

The simulations can be distinguished into __trainable__ and __fixed__ (or non-trainable) encodings for quantum learning models.
In particular, a sperate training of the two followed by numerical simulations to obtain robustness and generalization performances are performed.
Lastly, the results are plotted into a common plot for the generalization and robustness, respectively.

### Training

The training can be performed via running

``` bash
poetry run training_trainable
```

or

``` bash
poetry run training_non_trainable
```

respectively, from the root of the repository.

> __Note__: The training is parallelised using [Dask](https://www.dask.org), but it can still take a long time to finish.

The output of the training is stored in the __data__ directory.

### Generalization Simulations

Each of the two models can be used to perform simulations to obtain the corresponding generalization performance.
To do so, run

``` bash
poetry run analyse_generalization_trainable
```

or

``` bash
poetry run analyse_generalization_non_trainable
```

respectively.

> __Note__: The generalization simulations are parallelised with [Dask](https://www.dask.org) as well and can take up to several minutes.

The output of the generalization simulations are stored in the __data__ directory.

### Robustness Simulations

Each of the two models can also be used to perform simulations to obtain its robustness performance.
Therefore, run

``` bash
poetry run analyse_robustness_trainable
```

or

``` bash
poetry run analyse_robustness_non_trainable
```

respectively.

> __Note__: The robustness simulations are parallelised with [Dask](https://www.dask.org) as well and can take up to several minutes.

The output of the robustness simulations are stored in the __data__ directory.

### Visualizing the Results

The following commands can be run to visualize the results:

`poetry run plot_circuits` - To visualize the considered quantum circuits of the QML models.

`poetry run plot_generalization_trainable` - To visualize the generalization performance of the trainable encoding.

`poetry run plot_generalization_non_trainable` - To visualize the generalization performance of the fixed encoding.

`poetry run plot_robustness` - To create a common robustness plot for both QML models.

`poetry run plot_predictions` - To create a visualization of the considered classification problem: the *circle classification problem*.