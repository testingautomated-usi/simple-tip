# Reproduction Package of ISSTA 2022 paper "Simple Techniques Work Surprisingly Well..."

This is the reproduction package of the paper **Simple Techniques Work Surprisingly Well for Neural Network
Test Prioritization and Active Learning** by M.Weiss and P.Tonella, published at ISSTA 2022.

⚠️ This repository is indended for reproduction of our results; dependencies are not updated on purpose. The repository may thus contain vulnerabilities. We recommend you run it in a safe environment ⚠️

## Getting Started: Running the Reproduction Package

### Dependencies
On your machine, you'll need the following requirements:
- Docker
- Download the `assets.zip` folder from Zenodo ([link](https://doi.org/10.5281/zenodo.6504906)) and uncompress it. 
  For the remainder of this README, we will refer with `/path/to/assets/` as the path to your assets folder.
  *Note:* Due to the large size of our study, this achive is large: 
  Uncompressing it will take a while, and requires 8.3 GB of disk space. On windows, uncompressing is much faster when using [7zip](https://www.7-zip.org/download.html).
 

### Step 1: Running the container
Start the container with the following command (replacing `/path/to/assets/` with the path to the assets folder):
> docker run -it --rm -v /path/to/assets/:/assets ghcr.io/testingautomated-usi/simple-tip:latest

This will download and interactively run the docker image.

<sub>
Note: If running on linux with an nvidia-gpu, optionally install the [nvidia-docker toolkit](https://github.com/NVIDIA/nvidia-docker)
  which will allow you to use a GPU for training and inference.
  Then, add `--gpus all` after the `--rm` flag.
</sub>

You should now see a Tensorflow welcome message.

*Verify* that you mounted the `assets` volume successfully, by running `ls /assets`. You should see five folders (active_learning,  models,  priorities,  results,  times)


### Step 2: Running the reproduction package CLI

You can reproduce the results of the paper by using our provided command line interface as follows:

> python reproduction.py

Run `python reproduction.py --help` for more information on the available commands.

**Attention: Running any commands will modify the contents of the `/path/to/assets/` folder.**

You can exit the docker container by entering `exit`.

## Testing full functionality

### Testing all claims in the paper

The results in the paper can be verified in four parts.
For all steps, start reproduction with `python reproduction.py --phase evaluation`.
You are then provided the choice of the four parts:
- `test_prio`: Evaluates the test prioritization (table 1 in paper)
- `active_learning`: Evaluates the active learning (table 2 in paper) *takes a while*
- `test_prio_statistics`: Statistics about the test prioritization (figure 3 in paper)
- `active_learning_statistics`: Statistics about the active learning (figure 4 in paper) *takes a while*

All results will be stored in your mounted folder, specifically in `/path/to/assets/results/`.

### Re-Generating assets
Running our full experiments, even on a machine with 64GB RAM, 12 cores and multiple GPUs, took us multiple weeks to run.
For reproducibility, we stored all the corresponding intermediate results (e.g. models, priorities, ...) in the assets folder.
The evaluations explained above are working on these assets.

While running the full experiments is probably an overkill when assessing usability, you can 
of course start and of these steps by running `python reproduction.py` and choosing
any **other** phase than `evaluation`.

<sub>Please note that these steps contain tensorflow operations which are system dependent
and intrinsically random (that's why we conducted 100 re-runs) and that, if you abort any 
of these steps before completion, your assets folder might be in a corrupted state.
If in doubt, just replace the `/path/to/assets/` folder with a new one downloaded from zenodo.</sub>

To assess the reusability of our code, far beyond just reproducibility, 
we refer to the next section where we show dedicated, general-purpose artifacts
extracted from this reproduction package.

**AT Generation** Due to a 3rd party request after paper publication, 
we have added a new command `python reproduction.py --phase at_collection` to the CLI.
It allows to persist the activation traces for our models (all layers, including input and output)
to the file system. As above, the interactive CLI allows to narrow down the selection of datasets/models.
Attention: The ATs for all models and all dataset will add up to multiple terrabytes of data.
Running the AT generation command has no impact on the other experiments. 
Still, if anyone wants to reproduce using the exact version of the code we used, they should stick 
to the docker image and code for version `v0.1.0`.

## :rocket:  :rocket:  :rocket: Extracted General-Purpose Artifacts :rocket: :rocket: :rocket:
Running the above described reproduction package allows to verify the results 
shown in the paper.
We note however, that parts of our code might be directly usable in other contexts,
for which such a large and highly-complex reproduction package is 
not a suitable distribution format. 
We thus extracted, tested, documented and released three standalone artifacts of parts of our code,
which we expect to be particularly useful for other research projects.

### `dnn-tip` TIP implementations, and related utilities
About A collection of dnn test input prioritizers often used as benchmarks in recent literature.

Repository: [https://github.com/testingautomated-usi/dnn-tip](https://github.com/testingautomated-usi/dnn-tip)

PyPi: [https://pypi.org/project/dnn-tip/](https://pypi.org/project/dnn-tip/)

### `fashion-mnist-c` A corrupted dataset for fashion-mnist
A corrupted Fashion-MNIST benchmark for testing out-of-distribution robustness of computer vision models.

Repository: [https://github.com/testingautomated-usi/fashion-mnist-c](https://github.com/testingautomated-usi/fashion-mnist-c)

HuggingFace: [https://huggingface.co/datasets/mweiss/fashion_mnist_corrupted](https://huggingface.co/datasets/mweiss/fashion_mnist_corrupted)




### `corrupted-text` A text corruption utility, e.g. to generate imdb-c
A python library to generate out-of-distribution text datasets. Specifically, the library applies model-independent, commonplace corruptions (not model-specific, worst-case adversarial corruptions). We thus aim to allow benchmark-studies regarding robustness against realistic outliers.

Repository: [https://github.com/testingautomated-usi/corrupted-text](https://github.com/testingautomated-usi/corrupted-text)

PyPi: [https://pypi.org/project/corrupted-text/](https://pypi.org/project/corrupted-text/)


## Code in this repository

The following provides an overview of our python packages in the reproduction code:

- `./src/dnn_test_prio/`
Contains the modules to configure and run our experiments. Specifically, this includes one `runner` module per case study (e.g. case_study_imdb.py), whose main method allows to run the experiments for said case study. It also contains the specific selection of hyperparameters used for each case study. In addition, the package contains several generic modules (which are used by the `runner` modules) to create and use the appraoches implemented in the `core` package (e.g. handler_surprise.py).
- `./src/core/`
Contains the modules which we expect to be widely useful in 3rd party studies, much beyond the scope of our paper. It includes, e.g., the implementations of the tested approaches (surprise adequacies, neuron coverages, deep gini). It is designed to facilitate reusability: Special care was given to the code documentation, interfaces are defined using type-hints, and crutial hyperparameters are configurable. The core package is also largely tested by unit tests. Upon acceptance of our paper, the contents of this package will be publicly released as standalone pip-packages.
- `./src/plotters/`
Contains the logic to create the final results (tables, plots, etc.) from the raw intermediate results persisted when running the experiments. For example, the logic to average the results over all 100 re-runs, or the logic to compute the heatmap of effect size and p-values (Fig. 3 in the paper) can be found in this package.
- `./Dockerfile` and `./requirements.txt`
Specification of the software infrastructure we used to run the experiments. Building docker image specified in the docker file will create the same docker image as the one we used to run the experiments.
- `./reproduction.py`
A CLI which allows to easily re-run parts of our experiements (described above).

## Credits: 

- Thesaurus from https://github.com/zaibacu/thesaurus, free for commercial use, based on wordnet.
  Find license [here](https://wordnet.princeton.edu/license-and-commercial-use).
