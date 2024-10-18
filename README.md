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


GCN_t:.
│  baseline.py
│  extract.c
│  extract.cpython-38-x86_64-linux-gnu.so
│  extract.pxd
│  extract.pyx
│  kan_test.py
│  kan_train.py
│  README.md
│  setup.py
│  tempCodeRunnerFile.py
│  tree.txt
│  utilities.py
│  
├─atc
│  │  background.py
│  │  backgroundfixed.py
│  │  datagenerate.py
│  │  datageneratefixed.py
│  │  datagenerateremove.py
│  │  formula.py
│  │  linear.py
│  │  make_ini.py
│  │  nonlinear.py
│  │  predicate.py
│  │  
│  ├─stlpy
│  │  │  LISENCE
│  │  │  __init__.py
│  │  │  
│  │  ├─benchmarks
│  │  │  │  base.py
│  │  │  │  common.py
│  │  │  │  door_puzzle.py
│  │  │  │  either_or.py
│  │  │  │  narrow_passage.py
│  │  │  │  nonlinear_reach_avoid.py
│  │  │  │  random_multitarget.py
│  │  │  │  reach_avoid.py
│  │  │  │  stepping_stones.py
│  │  │  │  __init__.py
│  │  │  │  
│  │  │  └─__pycache__
│  │  │          
│  │  ├─solvers
│  │  │  │  base.py
│  │  │  │  __init__.py
│  │  │  │  
│  │  │  ├─drake
│  │  │  │  │  drake_base.py
│  │  │  │  │  drake_micp.py
│  │  │  │  │  drake_smooth.py
│  │  │  │  │  drake_sos1.py
│  │  │  │  │  __init__.py
│  │  │  │  │  
│  │  │  │  └─__pycache__
│  │  │  │          drake_base.cpython-310.pyc
│  │  │  │          drake_micp.cpython-310.pyc
│  │  │  │          drake_smooth.cpython-310.pyc
│  │  │  │          drake_sos1.cpython-310.pyc
│  │  │  │          __init__.cpython-310.pyc
│  │  │  │          
│  │  │  ├─gurobi
│  │  │  │  │  gurobi_micp.py
│  │  │  │  │  __init__.py
│  │  │  │  │  
│  │  │  │  └─__pycache__
│  │  │  │          
│  │  │  ├─scipy
│  │  │  │  │  gradient_solver.py
│  │  │  │  │  __init__.py
│  │  │  │  │  
│  │  │  │  └─__pycache__
│  │  │  │          
│  │  │  └─__pycache__
│  │  │          
│  │  ├─STL
│  │  │  │  formula.py
│  │  │  │  predicate.py
│  │  │  │  __init__.py
│  │  │  │  
│  │  │  └─__pycache__
│  │  │          
│  │  ├─systems
│  │  │  │  linear.py
│  │  │  │  nonlinear.py
│  │  │  │  __init__.py
│  │  │  │  
│  │  │  └─__pycache__
│  │  │          
│  │  └─__pycache__
│  │          
│  └─__pycache__
│          
├─build
│  └─lib.linux-x86_64-cpython-38
│          execute.cpython-38-x86_64-linux-gnu.so
│          
├─ddpg
│  │  core.py
│  │  corekan.py
│  │  ddpgclass.py
│  │  ddpgclasskan.py
│  │  kan.py
│  │  model.py
│  │  modelkan.py
│  │  noise.py
│  │  user_config.py
│  │  util.py
│  │  
│  ├─utils
│  │  │  logx.py
│  │  │  mpi_tools.py
│  │  │  run_utils.py
│  │  │  serialization_utils.py
│  │  │  
│  │  └─__pycache__
│  │          
│  └─__pycache__
│          
├─example_model
│  │  vars.pkl
│  │  
│  ├─pyt_save
│  │      model.pt
│  │      
│  └─test_results
│      │  test_step100s.csv
│      │  test_step10s.csv
│      │  test_step1s.csv
│      │  test_step20s.csv
│      │  test_step2s.csv
│      │  test_step3s.csv
│      │  test_step4s.csv
│      │  test_step50s.csv
│      │  test_step5s.csv
│      │  test_step6s.csv
│      │  test_step7s.csv
│      │  test_step8s.csv
│      │  test_step9s.csv
│      │  
│      └─.ipynb_checkpoints
│              
├─smallatc
│  ├─test
│  │      
│  ├─train
│  │      
│  └─validation
│          
└─__pycache__
