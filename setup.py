from setuptools import setup,Extension
from Cython.Build import cythonize
# import numpy
# extensions = [
#     Extension('execute_1',['t1.pyx'],
#         include_path=["/home/yunbo/anaconda3/envs/test/include/scip",
#                       "/home/yunbo/anaconda3/pkgs/pyscipopt-4.2.0-py38hfa26641_2/lib/python3.8/site-packages/pyscipopt",
#                       "/home/yunbo/anaconda3/pkgs/numpy-1.24.4-py38h59b608b_0/lib/python3.8/site-packages/numpy/core/include"],  # Adjust SCIP include path
#         # include_dirs = ["/home/yunbo/anaconda3/pkgs/pyscipopt-4.2.0-py38hfa26641_2/lib/python3.8/site-packages/pyscipopt"],
#         libraries=["scip"],  # This might need to be adjusted depending on how SCIP is installed
#         library_path=["/home/yunbo/anaconda3/envs/test/lib"],  # Adjust SCIP library path

#     )
# ]

extensions = [
    Extension('extract',['extract.pyx'],
              include_dirs=["/home/yunbo/anaconda3/envs/test/include/scip",
                            "/home/yunbo/anaconda3/pkgs/pyscipopt-4.2.0-py38hfa26641_2/lib/python3.8/site-packages/pyscipopt",
                            "/home/yunbo/anaconda3/pkgs/numpy-1.24.4-py38h59b608b_0/lib/python3.8/site-packages/numpy/core/include"
                            ],
              library_dirs=["/path/to/scip/lib"],
              libraries=["scip"],),
              

]
setup(
    ext_modules=cythonize(extensions,language_level=3),
)
# setup(ext_modules=cythonize("t1.pyx", language_level=3))