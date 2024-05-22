# GCN_t
## Installation
### Conda
`conda config --add channels conda-forge`

`conda config --add channels pytorch`

`conda create -n name python=3.8 pytorch scip=8.0.1 pyscipopt numpy`

`conda install pyg -c pyg`

`conda install cython`

Download the ecole package [here](https://drive.google.com/file/d/1vXdfIeeoCctlHszhg7n1goBcEs052A0q/view?usp=drive_link) and extract it.

`cd ecole_0.8.1`

To specify the where to find SCIP and install

`CMAKE_ARGS="-DSCIP_DIR=path/to/lib/cmake/scip -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON" python -m pip install .`

Compile command

`gcc -shared -fPIC -o execute.so execute.c -I/path/to/envs/envname/include -I/path/to/envs/envname/include/python3.8 -L/path/to/envs/envname/lib -lscip`
