# GCN_t
## Installation
### Operating System

Ubuntu 22.04

### Conda
`conda config --add channels conda-forge`

`conda config --add channels pytorch`

`conda create -n name python=3.8 pytorch scip=8.0.1 pyscipopt numpy treelib matplotlib`

`conda install cython scipy joblib mpi4py`

If you want to generate new instances, please install gurobi:

`conda install -c gurobi gurobi`

## Run
Before run the code, please compile the cython function: modify the path of `include_dirs` and `library_dirs` in `setup.py` to your own environment. Then, run the command

`python3 setup.py build_ext --inplace`

### Training

`python3 -u kan_train.py`

### Testing

`python3 -u kan_test.py`

### New Instances
To generate new instances, run `atc/datagenerateremove.py` (smaller size instances) or `atc/datageneratefixed.py` (larger size instances). In the report, we are using the smaller one (11590 variables*13000 constraints)

`python -u datagenerateremove.py`

Then, make the initial solution for new instances by using `atc/make_ini.py`

`python -u make_ini.py`

## Hierarchical structure


