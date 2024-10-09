# GCN_t
## Installation
### Conda
`conda config --add channels conda-forge`

`conda config --add channels pytorch`

`conda create -n name python=3.8 pytorch scip=8.0.1 pyscipopt numpy`

`conda install cython scipy joblib mpi4py`

If you want to generate new instances, please install gurobi:

`conda install -c gurobi gurobi`

## Run
To specify the where to find SCIP and install

`CMAKE_ARGS="-DSCIP_DIR=path/to/lib/cmake/scip -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON" python -m pip install .`

Compile command

`gcc -shared -fPIC -o execute.so execute.c -I/path/to/envs/envname/include -I/path/to/envs/envname/include/python3.8 -L/path/to/envs/envname/lib -lscip`
