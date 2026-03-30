
# Installation

```

To contribute or develop locally, use files as base repository and run:

```bash
cd gridfm-graphkit
python -m venv venv
source venv/bin/activate
pip install -e .
```

For documentation generation and unit testing, install with the optional `dev` and `test` extras:

```bash
pip install -e .[dev,test]
```

There is also an `requirements.txt` that can be used to generate the virtual environment to run all experiments. The data has to be downloaded seperately. In this dataset, only the unseen data is contained.

# Reproduce experiments 

First steps are to download the data from Zenodo. 
The folder containing all datasets is then added to 
gridFM-graphkit/examples/MI.py in the variable data_directory.



## Basic Steps
To reproduce the experiments, a two-step procedure is needed. The first step is always to geerate the data, in the seocnd step, the data is loaded and the membership classifier can be trained. In (almost) each step, there is thus a sweep iteration over
- data generation
- model12 

where the second serves to run for both the small and large model. To start these experiments, use `yamlRun.py --file Baseline.yaml` to obtain the baseline, for example.

As a last step, the file genResultsPaper.py is used to genrate the paper's figures.

## Generate data and train membership classifier
The following yaml files need to run, which are all except the baseline named after the corresponding sections.
- Baseline.yaml

Generates the baseline data and membershipo models.
- Exp5_2_2.yaml

Repeats the experiments, but using not output, but error.
- Exp5_2_3.yaml

This step reuse the generated data from the baselines, and runs an ablation study on the samples.
- Exp5_2_4Batch.yaml
- Exp5_2_4Node.yaml

This steps generates the data for the inference on a node or a batch.
- Exp5_2_5Zeros.yaml
- Exp5_2_5Ones.yaml

These generate the data based on replacing the original files with zero and ones, respectively.
- Exp5_2_6.yaml

Generates the data not based on all six features, but per features. The runtime is thus very slow, as the process runs six times. 

- Exp5_2_7.yaml

While the previous steps already contain the evaluation on two topologies and the generalization to the remaining data, this step generates the leave-1-out experiemnts, where one topology from seen and unssen data is left out and used to test generalization.

## Generate plots when experiments are comopleted
To obtain the plots and tables, simply run `python genResultsPaper.py`. 
If a specific table or plot is needed, simply comment the desired section within the python file.

## Explanantion of variable beyond experimental settings
The following variable can be configured for experiments:
 - **generate**: true, generate data for Membership; false: train membership classiifer
 - **model12**: true, small model; false, false model
 - **error_based**: false, based on output; true: based on error
 - **ablation**: false, standard setting; true, tests different training set sizes for membership
 - **randomMask**: false, uses standard mask; true, random mask (unused, not reasoanble for task studied)
 - **removeNan**: false, does not remove nan, true: remove nan; for better comptaibility with some classifiers
 - **graphWise**: true, based on topology, false, not based on topology
 - **nodeWise**: true, based on nodes, false, not based on nodes
 - if both grapWise and nodeWise are false, we default to prediciting batch-wise
 - **topologyOnly**: false, uses input features; true, replaces features with zeros
 - **ones**: true, replaces features with ones. Only effective together with topologyOnly=true
 - **modelComparison**: true, tests all models, false: tests only SVM 
 - **leaveOneOut**: false, trains on 2 datasets and tests generalization; true: tests on n-2 datasets and tests generalization