import weakref
from os.path import abspath
from os.path import splitext
import sys
import warnings
import numpy as np

cimport numpy as np
from cpython cimport Py_INCREF, Py_DECREF
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from libc.stdlib cimport malloc, free
from libc.stdio cimport fdopen
from numpy.math cimport INFINITY, NAN
from libc.math cimport sqrt as SQRT
from pyscipopt.scip import PY_SCIP_CALL


cdef class PY_SCIP_RESULT:
    DIDNOTRUN   = SCIP_DIDNOTRUN
    DELAYED     = SCIP_DELAYED
    DIDNOTFIND  = SCIP_DIDNOTFIND
    FEASIBLE    = SCIP_FEASIBLE
    INFEASIBLE  = SCIP_INFEASIBLE
    UNBOUNDED   = SCIP_UNBOUNDED
    CUTOFF      = SCIP_CUTOFF
    SEPARATED   = SCIP_SEPARATED
    NEWROUND    = SCIP_NEWROUND
    REDUCEDDOM  = SCIP_REDUCEDDOM
    CONSADDED   = SCIP_CONSADDED
    CONSCHANGED = SCIP_CONSCHANGED
    BRANCHED    = SCIP_BRANCHED
    SOLVELP     = SCIP_SOLVELP
    FOUNDSOL    = SCIP_FOUNDSOL
    SUSPENDED   = SCIP_SUSPENDED
    SUCCESS     = SCIP_SUCCESS
# cdef class extopt:
#     cdef SCIP*
def getState(model, prev_state = None):
    # PyCapsule_GetPointer
    cdef SCIP* scip = <SCIP*> PyCapsule_GetPointer(model, 'scip')
    cdef int i, j, k, col_i
    cdef SCIP_Real sim, prod

    update = prev_state is not None

    # COLUMNS
    # cdef pyscip.SCIP_COL** cols = pyscip.SCIPgetLPCols(scip)
    cdef SCIP_COL** cols = SCIPgetLPCols(scip)
    cdef int ncols = SCIPgetNLPCols(scip)

    cdef np.ndarray[np.int32_t,   ndim=1] col_types
    cdef np.ndarray[np.float32_t, ndim=1] col_coefs
    cdef np.ndarray[np.float32_t, ndim=1] col_lbs
    cdef np.ndarray[np.float32_t, ndim=1] col_ubs
    cdef np.ndarray[np.int32_t,   ndim=1] col_basestats
    cdef np.ndarray[np.float32_t, ndim=1] col_redcosts
    cdef np.ndarray[np.int32_t,   ndim=1] col_ages
    cdef np.ndarray[np.float32_t, ndim=1] col_solvals
    cdef np.ndarray[np.float32_t, ndim=1] col_solfracs
    cdef np.ndarray[np.int32_t,   ndim=1] col_sol_is_at_lb
    cdef np.ndarray[np.int32_t,   ndim=1] col_sol_is_at_ub
    cdef np.ndarray[np.float32_t, ndim=1] col_incvals
    cdef np.ndarray[np.float32_t, ndim=1] col_avgincvals

    if not update:
        col_types        = np.empty(shape=(ncols, ), dtype=np.int32)
        col_coefs        = np.empty(shape=(ncols, ), dtype=np.float32)
        col_lbs          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ubs          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_basestats    = np.empty(shape=(ncols, ), dtype=np.int32)
        col_redcosts     = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ages         = np.empty(shape=(ncols, ), dtype=np.int32)
        col_solvals      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_solfracs     = np.empty(shape=(ncols, ), dtype=np.float32)
        col_sol_is_at_lb = np.empty(shape=(ncols, ), dtype=np.int32)
        col_sol_is_at_ub = np.empty(shape=(ncols, ), dtype=np.int32)
        col_incvals      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_avgincvals   = np.empty(shape=(ncols, ), dtype=np.float32)
    else:
        col_types        = prev_state['col']['types']
        col_coefs        = prev_state['col']['coefs']
        col_lbs          = prev_state['col']['lbs']
        col_ubs          = prev_state['col']['ubs']
        col_basestats    = prev_state['col']['basestats']
        col_redcosts     = prev_state['col']['redcosts']
        col_ages         = prev_state['col']['ages']
        col_solvals      = prev_state['col']['solvals']
        col_solfracs     = prev_state['col']['solfracs']
        col_sol_is_at_lb = prev_state['col']['sol_is_at_lb']
        col_sol_is_at_ub = prev_state['col']['sol_is_at_ub']
        col_incvals      = prev_state['col']['incvals']
        col_avgincvals   = prev_state['col']['avgincvals']

    cdef SCIP_SOL* sol = SCIPgetBestSol(scip)
    cdef SCIP_VAR* var
    cdef SCIP_Real lb, ub, solval
    for i in range(ncols):
        col_i = SCIPcolGetLPPos(cols[i])
        var = SCIPcolGetVar(cols[i])

        lb = SCIPcolGetLb(cols[i])
        ub = SCIPcolGetUb(cols[i])
        solval = SCIPcolGetPrimsol(cols[i])

        if not update:
            # Variable type
            col_types[col_i] = SCIPvarGetType(var)

            # Objective coefficient
            col_coefs[col_i] = SCIPcolGetObj(cols[i])

        # Lower bound
        if SCIPisInfinity(scip, REALABS(lb)):
            col_lbs[col_i] = NAN
        else:
            col_lbs[col_i] = lb

        # Upper bound
        if SCIPisInfinity(scip, REALABS(ub)):
            col_ubs[col_i] = NAN
        else:
            col_ubs[col_i] = ub

        # Basis status
        col_basestats[col_i] = SCIPcolGetBasisStatus(cols[i])

        # Reduced cost
        col_redcosts[col_i] = SCIPgetColRedcost(scip, cols[i])

        # Age
        col_ages[col_i] = cols[i].age

        # LP solution value
        col_solvals[col_i] = solval
        col_solfracs[col_i] = SCIPfeasFrac(scip, solval)
        col_sol_is_at_lb[col_i] = SCIPisEQ(scip, solval, lb)
        col_sol_is_at_ub[col_i] = SCIPisEQ(scip, solval, ub)

        # Incumbent solution value
        if sol is NULL:
            col_incvals[col_i] = NAN
            col_avgincvals[col_i] = NAN
        else:
            col_incvals[col_i] = SCIPgetSolVal(scip, sol, var)
            col_avgincvals[col_i] = SCIPvarGetAvgSol(var)


    # ROWS
    cdef int nrows = SCIPgetNLPRows(scip)
    cdef SCIP_ROW** rows = SCIPgetLPRows(scip)
    # cdef SCIP_ROW** rows = SCIPgetLPRows(scip)

    cdef np.ndarray[np.float32_t, ndim=1] row_lhss
    cdef np.ndarray[np.float32_t, ndim=1] row_rhss
    cdef np.ndarray[np.int32_t,   ndim=1] row_nnzrs
    cdef np.ndarray[np.float32_t, ndim=1] row_dualsols
    cdef np.ndarray[np.int32_t,   ndim=1] row_basestats
    cdef np.ndarray[np.int32_t,   ndim=1] row_ages
    cdef np.ndarray[np.float32_t, ndim=1] row_activities
    cdef np.ndarray[np.float32_t, ndim=1] row_objcossims
    cdef np.ndarray[np.float32_t, ndim=1] row_norms
    cdef np.ndarray[np.int32_t,   ndim=1] row_is_at_lhs
    cdef np.ndarray[np.int32_t,   ndim=1] row_is_at_rhs

    if not update:
        row_lhss          = np.empty(shape=(nrows, ), dtype=np.float32)
        row_rhss          = np.empty(shape=(nrows, ), dtype=np.float32)
        row_nnzrs         = np.empty(shape=(nrows, ), dtype=np.int32)
        row_dualsols      = np.empty(shape=(nrows, ), dtype=np.float32)
        row_basestats     = np.empty(shape=(nrows, ), dtype=np.int32)
        row_ages          = np.empty(shape=(nrows, ), dtype=np.int32)
        row_activities    = np.empty(shape=(nrows, ), dtype=np.float32)
        row_objcossims    = np.empty(shape=(nrows, ), dtype=np.float32)
        row_norms         = np.empty(shape=(nrows, ), dtype=np.float32)
        row_is_at_lhs     = np.empty(shape=(nrows, ), dtype=np.int32)
        row_is_at_rhs     = np.empty(shape=(nrows, ), dtype=np.int32)
        row_is_local      = np.empty(shape=(nrows, ), dtype=np.int32)
        row_is_modifiable = np.empty(shape=(nrows, ), dtype=np.int32)
        row_is_removable  = np.empty(shape=(nrows, ), dtype=np.int32)
    else:
        row_lhss          = prev_state['row']['lhss']
        row_rhss          = prev_state['row']['rhss']
        row_nnzrs         = prev_state['row']['nnzrs']
        row_dualsols      = prev_state['row']['dualsols']
        row_basestats     = prev_state['row']['basestats']
        row_ages          = prev_state['row']['ages']
        row_activities    = prev_state['row']['activities']
        row_objcossims    = prev_state['row']['objcossims']
        row_norms         = prev_state['row']['norms']
        row_is_at_lhs     = prev_state['row']['is_at_lhs']
        row_is_at_rhs     = prev_state['row']['is_at_rhs']
        row_is_local      = prev_state['row']['is_local']
        row_is_modifiable = prev_state['row']['is_modifiable']
        row_is_removable  = prev_state['row']['is_removable']

    cdef int nnzrs = 0
    cdef SCIP_Real activity, lhs, rhs, cst
    for i in range(nrows):

        # lhs <= activity + cst <= rhs
        lhs = SCIProwGetLhs(rows[i])
        rhs = SCIProwGetRhs(rows[i])
        cst = SCIProwGetConstant(rows[i])
        activity = SCIPgetRowLPActivity(scip, rows[i])  # cst is part of activity

        if not update:
            # number of coefficients
            row_nnzrs[i] = SCIProwGetNLPNonz(rows[i])
            nnzrs += row_nnzrs[i]

            # left-hand-side
            if SCIPisInfinity(scip, REALABS(lhs)):
                row_lhss[i] = NAN
            else:
                row_lhss[i] = lhs - cst

            # right-hand-side
            if SCIPisInfinity(scip, REALABS(rhs)):
                row_rhss[i] = NAN
            else:
                row_rhss[i] = rhs - cst

            # row properties
            row_is_local[i] = SCIProwIsLocal(rows[i])
            row_is_modifiable[i] = SCIProwIsModifiable(rows[i])
            row_is_removable[i] = SCIProwIsRemovable(rows[i])

            # Objective cosine similarity - inspired from SCIProwGetObjParallelism()
            SCIPlpRecalculateObjSqrNorm(scip.set, scip.lp)
            prod = rows[i].sqrnorm * scip.lp.objsqrnorm
            row_objcossims[i] = rows[i].objprod / SQRT(prod) if SCIPisPositive(scip, prod) else 0.0

            # L2 norm
            row_norms[i] = SCIProwGetNorm(rows[i])  # cst ?

        # Dual solution
        row_dualsols[i] = SCIProwGetDualsol(rows[i])

        # Basis status
        row_basestats[i] = SCIProwGetBasisStatus(rows[i])

        # Age
        row_ages[i] = SCIProwGetAge(rows[i])

        # Activity
        row_activities[i] = activity - cst

        # Is tight
        row_is_at_lhs[i] = SCIPisEQ(scip, activity, lhs)
        row_is_at_rhs[i] = SCIPisEQ(scip, activity, rhs)


    cdef np.ndarray[np.int32_t,   ndim=1] coef_colidxs
    cdef np.ndarray[np.int32_t,   ndim=1] coef_rowidxs
    cdef np.ndarray[np.float32_t, ndim=1] coef_vals

    # Row coefficients
    if not update:
        coef_colidxs = np.empty(shape=(nnzrs, ), dtype=np.int32)
        coef_rowidxs = np.empty(shape=(nnzrs, ), dtype=np.int32)
        coef_vals    = np.empty(shape=(nnzrs, ), dtype=np.float32)
    else:
        coef_colidxs = prev_state['nzrcoef']['colidxs']
        coef_rowidxs = prev_state['nzrcoef']['rowidxs']
        coef_vals    = prev_state['nzrcoef']['vals']

    # cdef pyscip.SCIP_COL ** row_cols
    cdef SCIP_COL ** row_cols
    cdef SCIP_Real * row_vals

    if not update:
        j = 0
        for i in range(nrows):

            # coefficient indexes and values
            row_cols = SCIProwGetCols(rows[i])
            row_vals = SCIProwGetVals(rows[i])
            for k in range(row_nnzrs[i]):
                coef_colidxs[j+k] = SCIPcolGetLPPos(row_cols[k])
                coef_rowidxs[j+k] = i
                coef_vals[j+k] = row_vals[k]

            j += row_nnzrs[i]

    # free(scip)
    return {
        'col': {
            'types':        col_types,
            'coefs':        col_coefs,
            'lbs':          col_lbs,
            'ubs':          col_ubs,
            'basestats':    col_basestats,
            'redcosts':     col_redcosts,
            'ages':         col_ages,
            'solvals':      col_solvals,
            'solfracs':     col_solfracs,
            'sol_is_at_lb': col_sol_is_at_lb,
            'sol_is_at_ub': col_sol_is_at_ub,
            'incvals':      col_incvals,
            'avgincvals':   col_avgincvals,
        },
        'row': {
            'lhss':          row_lhss,
            'rhss':          row_rhss,
            'nnzrs':         row_nnzrs,
            'dualsols':      row_dualsols,
            'basestats':     row_basestats,
            'ages':          row_ages,
            'activities':    row_activities,
            'objcossims':    row_objcossims,
            'norms':         row_norms,
            'is_at_lhs':     row_is_at_lhs,
            'is_at_rhs':     row_is_at_rhs,
            'is_local':      row_is_local,
            'is_modifiable': row_is_modifiable,
            'is_removable':  row_is_removable,
        },
        'nzrcoef': {
            'colidxs': coef_colidxs,
            'rowidxs': coef_rowidxs,
            'vals':    coef_vals,
        },
        'stats': {
            'nlps': SCIPgetNLPs(scip)
        }
    }

def executeBranchRule(model, str name, allowaddcons):
    cdef SCIP* scip = <SCIP*> PyCapsule_GetPointer(model, 'scip')
    cdef SCIP_BRANCHRULE*  branchrule
    cdef SCIP_RESULT result
    branchrule = SCIPfindBranchrule(scip, name.encode("UTF-8"))
    if branchrule == NULL:
        print("Error, branching rule not found!")
        # free(scip)
        return PY_SCIP_RESULT.DIDNOTFIND
    else:
        branchrule.branchexeclp(scip, branchrule, allowaddcons, &result)
        # free(scip)
        return result

def getVartype(model, prev_state = None):
    cdef SCIP* scip = <SCIP*> PyCapsule_GetPointer(model, 'scip')
    cdef SCIP_VAR* transformed_var
    cdef SCIP_VAR** vars
    cdef int nvar,i
    update = prev_state is not None
    nvar = SCIPgetNOrigVars(scip)
    vars = SCIPgetOrigVars(scip)
    cdef np.ndarray[np.int32_t,   ndim=1] col_types
    cdef np.ndarray[np.float32_t, ndim=1] col_coefs
    cdef SCIP_Real lb, ub, solval
    cdef SCIP_SOL* sol = SCIPgetBestSol(scip)
    cdef SCIP_COL* col 
    if not update:
        col_types        = np.empty(shape=(nvar, ), dtype=np.int32)
        col_coefs        = np.empty(shape=(nvar, ), dtype=np.float32)
        col_lbs          = np.empty(shape=(nvar, ), dtype=np.float32)
        col_ubs          = np.empty(shape=(nvar, ), dtype=np.float32)
        col_basestats    = np.empty(shape=(nvar, ), dtype=np.int32)
        col_redcosts     = np.empty(shape=(nvar, ), dtype=np.float32)
        col_ages         = np.empty(shape=(nvar, ), dtype=np.int32)
        col_solvals      = np.empty(shape=(nvar, ), dtype=np.float32)
        col_solfracs     = np.empty(shape=(nvar, ), dtype=np.float32)
        col_sol_is_at_lb = np.empty(shape=(nvar, ), dtype=np.int32)
        col_sol_is_at_ub = np.empty(shape=(nvar, ), dtype=np.int32)
        col_incvals      = np.empty(shape=(nvar, ), dtype=np.float32)
        col_avgincvals   = np.empty(shape=(nvar, ), dtype=np.float32)
    else:
        col_types        = prev_state['col']['types']
        col_coefs        = prev_state['col']['coefs']
        col_lbs          = prev_state['col']['lbs']
        col_ubs          = prev_state['col']['ubs']
        col_basestats    = prev_state['col']['basestats']
        col_redcosts     = prev_state['col']['redcosts']
        col_ages         = prev_state['col']['ages']
        col_solvals      = prev_state['col']['solvals']
        col_solfracs     = prev_state['col']['solfracs']
        col_sol_is_at_lb = prev_state['col']['sol_is_at_lb']
        col_sol_is_at_ub = prev_state['col']['sol_is_at_ub']
        col_incvals      = prev_state['col']['incvals']
        col_avgincvals   = prev_state['col']['avgincvals']
    for i in range(nvar):
        # ptr = <size_t>(vars[i])
        lb = SCIPvarGetLbLocal(vars[i])
        ub = SCIPvarGetLbLocal(vars[i])
        
        solval = SCIPgetSolVal(scip,sol,vars[i])
        if not update:
            # Variable type
            col_types[i] = SCIPvarGetType(vars[i])

            # Objective coefficient
            col_coefs[i] = SCIPvarGetObj(vars[i])
        
        # Lower bound
        if SCIPisInfinity(scip, REALABS(lb)):
            col_lbs[i] = NAN
        else:
            col_lbs[i] = lb
        
        # Upper bound
        if SCIPisInfinity(scip, REALABS(ub)):
            col_ubs[i] = NAN
        else:
            col_ubs[i] = ub
        
        # Basis status
        if solval == lb: 
            col_basestats[i] = SCIP_BASESTAT_LOWER
        elif solval > lb and solval < ub:
            col_basestats[i] = SCIP_BASESTAT_BASIC
        elif solval == ub:
            col_basestats[i] = SCIP_BASESTAT_UPPER
        else:
            col_basestats[i] = SCIP_BASESTAT_ZERO
        
        # Reduced cost
        col_redcosts[i] = SCIPgetVarRedcost(scip, vars[i])

        # Age
        SCIPgetTransformedVar(scip, vars[i],&transformed_var)
        col = SCIPvarGetCol(transformed_var)
        col_ages[i] = col.age

        col_solvals[i] = solval
    return {
        'col':{
            'types':        col_types,
            'coefs':        col_coefs,
            'lbs':          col_lbs,
            'ubs':          col_ubs,
            'basestats':    col_basestats,
            'redcosts':     col_redcosts,
            'ages':         col_ages,
            'solvals':      col_solvals,
            'solfracs':     col_solfracs,
            'sol_is_at_lb': col_sol_is_at_lb,
            'sol_is_at_ub': col_sol_is_at_ub,
            'incvals':      col_incvals,
            'avgincvals':   col_avgincvals,
        },
    }


def test(model):
    cdef SCIP* scip = <SCIP*> PyCapsule_GetPointer(model, 'scip')
    onvar = SCIPgetNOrigVars(scip)
    cnvar = SCIPgetNVars(scip)
    return(onvar,cnvar)


def getAtcState(model, prev_state = None):
    # PyCapsule_GetPointer
    cdef SCIP* scip = <SCIP*> PyCapsule_GetPointer(model, 'scip')
    cdef int i, j, k, col_i
    cdef SCIP_Real sim, prod

    update = prev_state is not None

    # COLUMNS
    # cdef pyscip.SCIP_COL** cols = pyscip.SCIPgetLPCols(scip)
    cdef SCIP_COL** cols = SCIPgetLPCols(scip)
    # cdef int ncols = SCIPgetNLPCols(scip)
    cdef int ncols = SCIPgetNOrigVars(scip)

    cdef np.ndarray[np.int32_t,   ndim=1] col_types
    cdef np.ndarray[np.float32_t, ndim=1] col_coefs
    cdef np.ndarray[np.float32_t, ndim=1] col_lbs
    cdef np.ndarray[np.float32_t, ndim=1] col_ubs
    cdef np.ndarray[np.int32_t,   ndim=1] col_basestats
    cdef np.ndarray[np.float32_t, ndim=1] col_redcosts
    cdef np.ndarray[np.int32_t,   ndim=1] col_ages
    cdef np.ndarray[np.float32_t, ndim=1] col_solvals
    cdef np.ndarray[np.float32_t, ndim=1] col_solfracs
    cdef np.ndarray[np.int32_t,   ndim=1] col_sol_is_at_lb
    cdef np.ndarray[np.int32_t,   ndim=1] col_sol_is_at_ub
    cdef np.ndarray[np.float32_t, ndim=1] col_incvals
    cdef np.ndarray[np.float32_t, ndim=1] col_avgincvals

    if not update:
        col_types        = np.empty(shape=(ncols, ), dtype=np.int32)
        col_coefs        = np.empty(shape=(ncols, ), dtype=np.float32)
        col_lbs          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ubs          = np.empty(shape=(ncols, ), dtype=np.float32)
        col_basestats    = np.empty(shape=(ncols, ), dtype=np.int32)
        col_redcosts     = np.empty(shape=(ncols, ), dtype=np.float32)
        col_ages         = np.empty(shape=(ncols, ), dtype=np.int32)
        col_solvals      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_solfracs     = np.empty(shape=(ncols, ), dtype=np.float32)
        col_sol_is_at_lb = np.empty(shape=(ncols, ), dtype=np.int32)
        col_sol_is_at_ub = np.empty(shape=(ncols, ), dtype=np.int32)
        col_incvals      = np.empty(shape=(ncols, ), dtype=np.float32)
        col_avgincvals   = np.empty(shape=(ncols, ), dtype=np.float32)
    else:
        col_types        = prev_state['col']['types']
        col_coefs        = prev_state['col']['coefs']
        col_lbs          = prev_state['col']['lbs']
        col_ubs          = prev_state['col']['ubs']
        col_basestats    = prev_state['col']['basestats']
        col_redcosts     = prev_state['col']['redcosts']
        col_ages         = prev_state['col']['ages']
        col_solvals      = prev_state['col']['solvals']
        col_solfracs     = prev_state['col']['solfracs']
        col_sol_is_at_lb = prev_state['col']['sol_is_at_lb']
        col_sol_is_at_ub = prev_state['col']['sol_is_at_ub']
        col_incvals      = prev_state['col']['incvals']
        col_avgincvals   = prev_state['col']['avgincvals']

    cdef SCIP_SOL* sol = SCIPgetBestSol(scip)
    cdef SCIP_VAR* var 
    cdef SCIP_VAR* transformed_var
    cdef SCIP_Real lb, ub, solval
    cdef SCIP_VAR** vars = SCIPgetOrigVars(scip)
    cdef SCIP_COL* col
    
    for i in range(ncols):
        var = vars[i]
        SCIPgetTransformedVar(scip, var,&transformed_var)
        col = SCIPvarGetCol(transformed_var)
        col_i = SCIPcolGetLPPos(col)
        # var = SCIPcolGetVar(cols[i])

        lb = SCIPcolGetLb(col)
        ub = SCIPcolGetUb(col)
        solval = SCIPcolGetPrimsol(col)

        if not update:
            # Variable type
            col_types[col_i] = SCIPvarGetType(var)

            # Objective coefficient
            col_coefs[col_i] = SCIPcolGetObj(col)

        # Lower bound
        if SCIPisInfinity(scip, REALABS(lb)):
            col_lbs[col_i] = NAN
        else:
            col_lbs[col_i] = lb

        # Upper bound
        if SCIPisInfinity(scip, REALABS(ub)):
            col_ubs[col_i] = NAN
        else:
            col_ubs[col_i] = ub

        # Basis status
        col_basestats[col_i] = SCIPcolGetBasisStatus(col)

        # Reduced cost
        col_redcosts[col_i] = SCIPgetColRedcost(scip, col)

        # Age
        col_ages[col_i] = col.age

        # LP solution value
        col_solvals[col_i] = solval
        col_solfracs[col_i] = SCIPfeasFrac(scip, solval)
        col_sol_is_at_lb[col_i] = SCIPisEQ(scip, solval, lb)
        col_sol_is_at_ub[col_i] = SCIPisEQ(scip, solval, ub)

        # Incumbent solution value
        if sol is NULL:
            col_incvals[col_i] = NAN
            col_avgincvals[col_i] = NAN
        else:
            col_incvals[col_i] = SCIPgetSolVal(scip, sol, var)
            col_avgincvals[col_i] = SCIPvarGetAvgSol(var)


    # ROWS
    cdef int nrows = SCIPgetNLPRows(scip)

    cdef SCIP_ROW** rows = SCIPgetLPRows(scip)


    cdef np.ndarray[np.float32_t, ndim=1] row_lhss
    cdef np.ndarray[np.float32_t, ndim=1] row_rhss
    cdef np.ndarray[np.int32_t,   ndim=1] row_nnzrs
    cdef np.ndarray[np.float32_t, ndim=1] row_dualsols
    cdef np.ndarray[np.int32_t,   ndim=1] row_basestats
    cdef np.ndarray[np.int32_t,   ndim=1] row_ages
    cdef np.ndarray[np.float32_t, ndim=1] row_activities
    cdef np.ndarray[np.float32_t, ndim=1] row_objcossims
    cdef np.ndarray[np.float32_t, ndim=1] row_norms
    cdef np.ndarray[np.int32_t,   ndim=1] row_is_at_lhs
    cdef np.ndarray[np.int32_t,   ndim=1] row_is_at_rhs

    if not update:
        row_lhss          = np.empty(shape=(nrows, ), dtype=np.float32)
        row_rhss          = np.empty(shape=(nrows, ), dtype=np.float32)
        row_nnzrs         = np.empty(shape=(nrows, ), dtype=np.int32)
        row_dualsols      = np.empty(shape=(nrows, ), dtype=np.float32)
        row_basestats     = np.empty(shape=(nrows, ), dtype=np.int32)
        row_ages          = np.empty(shape=(nrows, ), dtype=np.int32)
        row_activities    = np.empty(shape=(nrows, ), dtype=np.float32)
        row_objcossims    = np.empty(shape=(nrows, ), dtype=np.float32)
        row_norms         = np.empty(shape=(nrows, ), dtype=np.float32)
        row_is_at_lhs     = np.empty(shape=(nrows, ), dtype=np.int32)
        row_is_at_rhs     = np.empty(shape=(nrows, ), dtype=np.int32)
        row_is_local      = np.empty(shape=(nrows, ), dtype=np.int32)
        row_is_modifiable = np.empty(shape=(nrows, ), dtype=np.int32)
        row_is_removable  = np.empty(shape=(nrows, ), dtype=np.int32)
    else:
        row_lhss          = prev_state['row']['lhss']
        row_rhss          = prev_state['row']['rhss']
        row_nnzrs         = prev_state['row']['nnzrs']
        row_dualsols      = prev_state['row']['dualsols']
        row_basestats     = prev_state['row']['basestats']
        row_ages          = prev_state['row']['ages']
        row_activities    = prev_state['row']['activities']
        row_objcossims    = prev_state['row']['objcossims']
        row_norms         = prev_state['row']['norms']
        row_is_at_lhs     = prev_state['row']['is_at_lhs']
        row_is_at_rhs     = prev_state['row']['is_at_rhs']
        row_is_local      = prev_state['row']['is_local']
        row_is_modifiable = prev_state['row']['is_modifiable']
        row_is_removable  = prev_state['row']['is_removable']

    cdef int nnzrs = 0
    cdef SCIP_ROW* row
    cdef SCIP_Real activity, lhs, rhs, cst
    cdef SCIP_CONS* cons
    # cdef SCIP_CONS* transformed_cons
    # cdef SCIP_CONS** conss = SCIPgetOrigConss(scip)
    cdef SCIP_CONS** conss = SCIPgetConss(scip)
    cdef SCIP_CONSHDLR** conshdlrs = SCIPgetConshdlrs(scip)
    cdef SCIP_CONSHDLR* conshdlr
    # cdef SCIP_CONS** SCIPconshdlrGetConss()
    cdef int temp
    cdef SCIP_Bool success
    # for i in range(nrows):
    #     cons = conss[i]
    #     conshdlr = conshdlrs[i]
    #     transformed_cons = SCIPconsGetTransformed(cons)
    #     # SCIPgetTransformedCons(scip,cons,&transformed_cons)
    #     row = SCIPconsGetRow(scip, transformed_cons)
    #     # row = SCIPconsGetRow(scip, cons)
    #     # lhs <= activity + cst <= rhs
    #     lhs = SCIProwGetLhs(row)
    #     rhs = SCIProwGetRhs(row)
    #     cst = SCIProwGetConstant(row)
    #     # activity = SCIPgetRowLPActivity(scip, row)  # cst is part of activity

    #     if not update:
    #     #     # number of coefficients
    #         PY_SCIP_CALL(SCIPconsGetNVars(cons, scip.set, &temp, &success))
    #         # row_nnzrs[i] = SCIProwGetNLPNonz(row)
    #         row_nnzrs[i] = temp
    #         nnzrs += row_nnzrs[i]

    #         # left-hand-side
    #         # if SCIPisInfinity(scip, REALABS(lhs)):
    #         #     row_lhss[i] = NAN
    #         # else:
    #         #     row_lhss[i] = lhs - cst

    #         # # right-hand-side
    #         # if SCIPisInfinity(scip, REALABS(rhs)):
    #         #     row_rhss[i] = NAN
    #         # else:
    #         #     row_rhss[i] = rhs - cst

    #     #     # row properties
    #     #     row_is_local[i] = SCIProwIsLocal(row)
    #     #     row_is_modifiable[i] = SCIProwIsModifiable(row)
    #     #     row_is_removable[i] = SCIProwIsRemovable(row)

    #     #     # Objective cosine similarity - inspired from SCIProwGetObjParallelism()
    #     #     SCIPlpRecalculateObjSqrNorm(scip.set, scip.lp)
    #     #     prod = row.sqrnorm * scip.lp.objsqrnorm
    #     #     row_objcossims[i] = row.objprod / SQRT(prod) if SCIPisPositive(scip, prod) else 0.0

    #     #     # L2 norm
    #         # row_norms[i] = SCIProwGetNorm(row)  # cst ?

    #     # # Dual solution
    #     # row_dualsols[i] = SCIProwGetDualsol(row)

    #     # # Basis status
    #     # row_basestats[i] = SCIProwGetBasisStatus(row)

    #     # # Age
    #     # row_ages[i] = SCIProwGetAge(row)
    # cdef SCIP_CONS* row_cons
    for i in range(nrows):

        # lhs <= activity + cst <= rhs
        lhs = SCIProwGetLhs(rows[i])
        rhs = SCIProwGetRhs(rows[i])
        cst = SCIProwGetConstant(rows[i])
        activity = SCIPgetRowLPActivity(scip, rows[i])  # cst is part of activity

        if not update:
            # number of coefficients
            cons = conss[i]
            # PY_SCIP_CALL(SCIPconsGetNVars(cons, scip.set, &temp, &success))
            # row_nnzrs[i] = temp
            row_nnzrs[i] = SCIProwGetNLPNonz(rows[i])
            nnzrs += row_nnzrs[i]

            # left-hand-side
            if SCIPisInfinity(scip, REALABS(lhs)):
                row_lhss[i] = NAN
            else:
                row_lhss[i] = lhs - cst

            # right-hand-side
            if SCIPisInfinity(scip, REALABS(rhs)):
                row_rhss[i] = NAN
            else:
                row_rhss[i] = rhs - cst

            # row properties
            row_is_local[i] = SCIProwIsLocal(rows[i])
            row_is_modifiable[i] = SCIProwIsModifiable(rows[i])
            row_is_removable[i] = SCIProwIsRemovable(rows[i])

            # Objective cosine similarity - inspired from SCIProwGetObjParallelism()
            SCIPlpRecalculateObjSqrNorm(scip.set, scip.lp)
            prod = rows[i].sqrnorm * scip.lp.objsqrnorm
            row_objcossims[i] = rows[i].objprod / SQRT(prod) if SCIPisPositive(scip, prod) else 0.0

            # L2 norm
            row_norms[i] = SCIProwGetNorm(rows[i])  # cst ?

        # Dual solution
        row_dualsols[i] = SCIProwGetDualsol(rows[i])

        # Basis status
        row_basestats[i] = SCIProwGetBasisStatus(rows[i])

        # Age
        row_ages[i] = SCIProwGetAge(rows[i])

        # Activity
        row_activities[i] = activity - cst

        # Is tight
        row_is_at_lhs[i] = SCIPisEQ(scip, activity, lhs)
        row_is_at_rhs[i] = SCIPisEQ(scip, activity, rhs)

    cdef np.ndarray[np.int32_t,   ndim=1] coef_colidxs
    cdef np.ndarray[np.int32_t,   ndim=1] coef_rowidxs
    cdef np.ndarray[np.float32_t, ndim=1] coef_vals

    # Row coefficients
    if not update:
        coef_colidxs = np.empty(shape=(nnzrs, ), dtype=np.int32)
        coef_rowidxs = np.empty(shape=(nnzrs, ), dtype=np.int32)
        coef_vals    = np.empty(shape=(nnzrs, ), dtype=np.float32)
    else:
        coef_colidxs = prev_state['nzrcoef']['colidxs']
        coef_rowidxs = prev_state['nzrcoef']['rowidxs']
        coef_vals    = prev_state['nzrcoef']['vals']

    # # cdef pyscip.SCIP_COL ** row_cols
    cdef SCIP_COL ** row_cols
    cdef SCIP_Real * row_vals
    cdef SCIP_VAR** row_vars
    cdef int i_1
    cdef SCIP_Bool success_2
    cdef SCIP_VAR* row_var
    cdef SCIP_CONS* row_cons
    if not update:
        j = 0
        for i in range(nrows):

            # coefficient indexes and values
            row_cols = SCIProwGetCols(rows[i])
            # row_cons = SCIProwGetOriginCons(rows[i])
            # PY_SCIP_CALL(SCIPgetConsVals(scip,row_cons,row_vals,ncols,&success_2))
            row_vals = SCIProwGetVals(rows[i])
            for k in range(row_nnzrs[i]):
                # row_var = SCIPcolGetVar(row_cols[k])
                coef_colidxs[j+k] = SCIPcolGetVarProbindex(row_cols[k])
                # coef_colidxs[j+k] = SCIPcolGetLPPos(row_cols[k])
                # coef_colidxs[j+k] = SCIPvarGetIndex(row_var)
                coef_rowidxs[j+k] = i

                coef_vals[j+k] = row_vals[k]

            j += row_nnzrs[i]

    # free(scip)
    return {
        'col': {
            'types':        col_types,
            'coefs':        col_coefs,
            'lbs':          col_lbs,
            'ubs':          col_ubs,
            'basestats':    col_basestats,
            'redcosts':     col_redcosts,
            'ages':         col_ages,
            'solvals':      col_solvals,
            'solfracs':     col_solfracs,
            'sol_is_at_lb': col_sol_is_at_lb,
            'sol_is_at_ub': col_sol_is_at_ub,
            'incvals':      col_incvals,
            'avgincvals':   col_avgincvals,
        },
        'row': {
            'lhss':          row_lhss,
            'rhss':          row_rhss,
            'nnzrs':         row_nnzrs,
            # 'dualsols':      row_dualsols,
            # 'basestats':     row_basestats,
            # 'ages':          row_ages,
            # 'activities':    row_activities,
            # 'objcossims':    row_objcossims,
            'norms':         row_norms,
            # 'is_at_lhs':     row_is_at_lhs,
            # 'is_at_rhs':     row_is_at_rhs,
            # 'is_local':      row_is_local,
            # 'is_modifiable': row_is_modifiable,
            # 'is_removable':  row_is_removable,
        },
        'nzrcoef': {
            'colidxs': coef_colidxs,
            'rowidxs': coef_rowidxs,
            'vals':    coef_vals,
        },
        'stats': {
            'nlps': SCIPgetNLPs(scip)
        }
    }

