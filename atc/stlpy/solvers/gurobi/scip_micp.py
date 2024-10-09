from ..base import STLSolver
from ...STL import LinearPredicate, NonlinearPredicate
import numpy as np

import pyscipopt as scip

import time
class ScipMICPSolver(STLSolver):
    def __init__(self, spec, sys, x0, T, M=100000, robustness_cost=True, 
            presolve=True, verbose=True):
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, sys, x0, T, verbose)

        self.M = float(M)
        self.presolve = presolve

        self.model = scip.Model("STL_MICP")

        self.cost = 0.0

        if not self.presolve:
            self.model.setParam('presolving/maxrounds', 0)
        if not self.verbose:
            self.model.setIntParam('display/verblevel', 0)

        if self.verbose:
            print("Setting up optimization problem...")
            st = time.time()  # for computing setup time

        