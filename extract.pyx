import weakref
from os.path import abspath
from os.path import splitext
import sys
import warnings
import numpy as np

cimport numpy as np
from cpython cimport Py_INCREF, Py_DECREF
from cpython.pycapsule cimport PyCapsule_GetPointer
from libc.stdlib cimport malloc, free
from libc.stdio cimport fdopen
from numpy.math cimport INFINITY, NAN
from libc.math cimport sqrt as SQRT
cimport scipopt.scip as pyscip
# from scipopt.scip cimport 
# from scip cimport SCIP_RETCODE,SCIP_VARTYPE,SCIP_OBJSENSE,SCIP_BOUNDTYPE,SCIP_BOUNDCHGTYPE,SCIP_RESULT,SCIP_STATUS,SCIP_STAGE,SCIP_NODETYPE,SCIP_PARAMSETTING,SCIP_PARAMTYPE,SCIP_PARAMEMPHASIS,SCIP_PROPTIMING,SCIP_PRESOLTIMING
# include "scipopt/expr.pxi"
# include "scipopt/lp.pxi"
# include "scipopt/benders.pxi"
# include "scipopt/benderscut.pxi"
# include "scipopt/branchrule.pxi"
# include "scipopt/conshdlr.pxi"
# include "scipopt/event.pxi"
# include "scipopt/heuristic.pxi"
# include "scipopt/presol.pxi"
# include "scipopt/pricer.pxi"
# include "scipopt/propagator.pxi"
# include "scipopt/sepa.pxi"
# include "scipopt/relax.pxi"

# cdef extern from "scip/def.h":
#     pyscip.SCIP_Real REALABS(pyscip.SCIP_Real x)
# ctypedef struct SCIP_COL:
#     int age
# ctypedef struct SCIP_ROW:
#     pyscip.SCIP_Real objprod
#     pyscip.SCIP_Real sqrnorm
#     pass
# cdef extern from "scip/scip.h":
#     ctypedef struct SCIP:
#         SCIP_SET* set
#         SCIP_STAT* stat
#         SCIP_PROB * origprob
#         SCIP_PROB * transprob
#         SCIP_LP* lp

#     ctypedef struct SCIP_SET:
#         pass

#     ctypedef struct SCIP_PROB:
#         pass
#     ctypedef struct SCIP_COL:
#         int age
#     ctypedef struct SCIP_ROW:
#         pyscip.SCIP_Real objprod
#         pyscip.SCIP_Real sqrnorm
#         pass
#     ctypedef struct SCIP_VAR:
#         pass
#     ctypedef struct SCIP_LP:
#         pyscip.SCIP_Real objsqrnorm

#     # SCIP_Real SCIPvarGetAvgSol(SCIP_VAR* var)
# cdef extern from "scip/struct_stat.h":
#     ctypedef struct SCIP_STAT:
#         # SCIP_REGRESSION *     regressioncandsobjval
#         pyscip.SCIP_Longint    nlpiterations
#         pyscip.SCIP_Longint    nrootlpiterations
#         pyscip.SCIP_Longint    nrootfirstlpiterations
#         pyscip.SCIP_Longint    nprimallpiterations
#         pyscip.SCIP_Longint    nduallpiterations
#         pyscip.SCIP_Longint    nlexduallpiterations
#         pyscip.SCIP_Longint    nbarrierlpiterations
#         pyscip.SCIP_Longint    nprimalresolvelpiterations
#         pyscip.SCIP_Longint    ndualresolvelpiterations
#         pyscip.SCIP_Longint    nlexdualresolvelpiterations
#         pyscip.SCIP_Longint    nnodelpiterations
#         pyscip.SCIP_Longint    ninitlpiterations
#         pyscip.SCIP_Longint    ndivinglpiterations
#         pyscip.SCIP_Longint    ndivesetlpiterations
#         pyscip.SCIP_Longint    nsbdivinglpiterations
#         pyscip.SCIP_Longint    nsblpiterations
#         pyscip.SCIP_Longint    nrootsblpiterations
#         pyscip.SCIP_Longint    nconflictlpiterations
#         pyscip.SCIP_Longint    nnodes
#         pyscip.SCIP_Longint    ninternalnodes
#         pyscip.SCIP_Longint    nobjleaves
#         pyscip.SCIP_Longint    nfeasleaves
#         pyscip.SCIP_Longint    ninfeasleaves
#         pyscip.SCIP_Longint    ntotalnodes
#         pyscip.SCIP_Longint    ntotalinternalnodes
#         pyscip.SCIP_Longint    ntotalnodesmerged
#         pyscip.SCIP_Longint    ncreatednodes
#         pyscip.SCIP_Longint    ncreatednodesrun
#         pyscip.SCIP_Longint    nactivatednodes
#         pyscip.SCIP_Longint    ndeactivatednodes
#         pyscip.SCIP_Longint    nearlybacktracks
#         pyscip.SCIP_Longint    nnodesaboverefbound
#         pyscip.SCIP_Longint    nbacktracks
#         pyscip.SCIP_Longint    ndelayedcutoffs
#         pyscip.SCIP_Longint    nreprops
#         pyscip.SCIP_Longint    nrepropboundchgs
#         pyscip.SCIP_Longint    nrepropcutoffs
#         pyscip.SCIP_Longint    nlpsolsfound
#         pyscip.SCIP_Longint    nrelaxsolsfound
#         pyscip.SCIP_Longint    npssolsfound
#         pyscip.SCIP_Longint    nsbsolsfound
#         pyscip.SCIP_Longint    nlpbestsolsfound
#         pyscip.SCIP_Longint    nrelaxbestsolsfound
#         pyscip.SCIP_Longint    npsbestsolsfound
#         pyscip.SCIP_Longint    nsbbestsolsfound
#         pyscip.SCIP_Longint    nexternalsolsfound
#         pyscip.SCIP_Longint    lastdispnode
#         pyscip.SCIP_Longint    lastdivenode
#         pyscip.SCIP_Longint    lastconflictnode
#         pyscip.SCIP_Longint    bestsolnode
#         pyscip.SCIP_Longint    domchgcount
#         pyscip.SCIP_Longint    nboundchgs
#         pyscip.SCIP_Longint    nholechgs
#         pyscip.SCIP_Longint    nprobboundchgs
#         pyscip.SCIP_Longint    nprobholechgs
#         pyscip.SCIP_Longint    nsbdowndomchgs
#         pyscip.SCIP_Longint    nsbupdomchgs
#         pyscip.SCIP_Longint    nsbtimesiterlimhit
#         pyscip.SCIP_Longint    nnodesbeforefirst
#         pyscip.SCIP_Longint    ninitconssadded
#         pyscip.SCIP_Longint    nactiveconssadded
#         pyscip.SCIP_Longint    externmemestim
#         pyscip.SCIP_Real   avgnnz
#         pyscip.SCIP_Real   firstlpdualbound
#         pyscip.SCIP_Real   rootlowerbound
#         pyscip.SCIP_Real   vsidsweight
#         pyscip.SCIP_Real   firstprimalbound
#         pyscip.SCIP_Real   firstprimaltime
#         pyscip.SCIP_Real   firstsolgap
#         pyscip.SCIP_Real   lastsolgap
#         pyscip.SCIP_Real   primalzeroittime
#         pyscip.SCIP_Real   dualzeroittime
#         pyscip.SCIP_Real   barrierzeroittime
#         pyscip.SCIP_Real   maxcopytime
#         pyscip.SCIP_Real   mincopytime
#         pyscip.SCIP_Real   firstlptime
#         pyscip.SCIP_Real   lastbranchvalue
#         pyscip.SCIP_Real   primaldualintegral
#         pyscip.SCIP_Real   previousgap
#         pyscip.SCIP_Real   previntegralevaltime
#         pyscip.SCIP_Real   lastprimalbound
#         pyscip.SCIP_Real   lastdualbound
#         pyscip.SCIP_Real   lastlowerbound
#         pyscip.SCIP_Real   lastupperbound
#         pyscip.SCIP_Real   rootlpbestestimate
#         pyscip.SCIP_Real   referencebound
#         pyscip.SCIP_Real   bestefficacy
#         pyscip.SCIP_Real   minefficacyfac
#         pyscip.SCIP_Real   detertimecnt
#         pyscip.SCIP_CLOCK *    solvingtime
#         pyscip.SCIP_CLOCK *    solvingtimeoverall
#         pyscip.SCIP_CLOCK *    presolvingtime
#         pyscip.SCIP_CLOCK *    presolvingtimeoverall
#         pyscip.SCIP_CLOCK *    primallptime
#         pyscip.SCIP_CLOCK *    duallptime
#         pyscip.SCIP_CLOCK *    lexduallptime
#         pyscip.SCIP_CLOCK *    barrierlptime
#         pyscip.SCIP_CLOCK *    divinglptime
#         pyscip.SCIP_CLOCK *    strongbranchtime
#         pyscip.SCIP_CLOCK *    conflictlptime
#         pyscip.SCIP_CLOCK *    lpsoltime
#         pyscip.SCIP_CLOCK *    relaxsoltime
#         pyscip.SCIP_CLOCK *    pseudosoltime
#         pyscip.SCIP_CLOCK *    sbsoltime
#         pyscip.SCIP_CLOCK *    nodeactivationtime
#         pyscip.SCIP_CLOCK *    nlpsoltime
#         pyscip.SCIP_CLOCK *    copyclock
#         pyscip.SCIP_CLOCK *    strongpropclock
#         pyscip.SCIP_CLOCK *    reoptupdatetime
#         pyscip.SCIP_HISTORY *  glbhistory
#         pyscip.SCIP_HISTORY *  glbhistorycrun
#         pyscip.SCIP_VAR *  lastbranchvar
#         # pyscip.SCIP_VISUAL *     visual
#         # pyscip.SCIP_HEUR *   firstprimalheur
#         pyscip.SCIP_STATUS     status
#         pyscip.SCIP_BRANCHDIR  lastbranchdir
#         pyscip.SCIP_LPSOLSTAT  lastsblpsolstats [2]
#         pyscip.SCIP_Longint    nnz
#         pyscip.SCIP_Longint    lpcount
#         pyscip.SCIP_Longint    relaxcount
#         pyscip.SCIP_Longint    nlps
#         pyscip.SCIP_Longint    nrootlps
#         pyscip.SCIP_Longint    nprimallps
#         pyscip.SCIP_Longint    nprimalzeroitlps
#         pyscip.SCIP_Longint    nduallps
#         pyscip.SCIP_Longint    ndualzeroitlps
#         pyscip.SCIP_Longint    nlexduallps
#         pyscip.SCIP_Longint    nbarrierlps
#         pyscip.SCIP_Longint    nbarrierzeroitlps
#         pyscip.SCIP_Longint    nprimalresolvelps
#         pyscip.SCIP_Longint    ndualresolvelps
#         pyscip.SCIP_Longint    nlexdualresolvelps
#         pyscip.SCIP_Longint    nnodelps
#         pyscip.SCIP_Longint    ninitlps
#         pyscip.SCIP_Longint    ndivinglps
#         pyscip.SCIP_Longint    ndivesetlps
#         pyscip.SCIP_Longint    nsbdivinglps
#         pyscip.SCIP_Longint    nnumtroublelpmsgs
#         pyscip.SCIP_Longint    nstrongbranchs
#         pyscip.SCIP_Longint    nrootstrongbranchs
#         pyscip.SCIP_Longint    nconflictlps
#         pyscip.SCIP_Longint    nnlps
#         pyscip.SCIP_Longint    nisstoppedcalls
#         pyscip.SCIP_Longint    totaldivesetdepth
#         int     subscipdepth
#         int     ndivesetcalls
#         int     nruns
#         int     ncutpoolfails
#         int     nconfrestarts
#         int     nrootboundchgs
#         int     nrootboundchgsrun
#         int     nrootintfixings
#         int     nrootintfixingsrun
#         int     prevrunnvars
#         int     nvaridx
#         int     ncolidx
#         int     nrowidx
#         int     marked_nvaridx
#         int     marked_ncolidx
#         int     marked_nrowidx
#         int     npricerounds
#         int     nseparounds
#         int     nincseparounds
#         int     ndisplines
#         int     maxdepth
#         int     maxtotaldepth
#         int     plungedepth
#         int     nactiveconss
#         int     nenabledconss
#         int     nimplications
#         int     npresolrounds
#         int     npresolroundsfast
#         int     npresolroundsmed
#         int     npresolroundsext
#         int     npresolfixedvars
#         int     npresolaggrvars
#         int     npresolchgvartypes
#         int     npresolchgbds
#         int     npresoladdholes
#         int     npresoldelconss
#         int     npresoladdconss
#         int     npresolupgdconss
#         int     npresolchgcoefs
#         int     npresolchgsides
#         int     lastnpresolfixedvars
#         int     lastnpresolaggrvars
#         int     lastnpresolchgvartypes
#         int     lastnpresolchgbds
#         int     lastnpresoladdholes
#         int     lastnpresoldelconss
#         int     lastnpresoladdconss
#         int     lastnpresolupgdconss
#         int     lastnpresolchgcoefs
#         int     lastnpresolchgsides
#         int     solindex
#         int     nrunsbeforefirst
#         int     firstprimaldepth
#         int     ncopies
#         int     nreoptruns
#         int     nclockskipsleft
#         pyscip.SCIP_Bool   memsavemode
#         pyscip.SCIP_Bool   userinterrupt
#         pyscip.SCIP_Bool   userrestart
#         pyscip.SCIP_Bool   inrestart
#         pyscip.SCIP_Bool   collectvarhistory
#         pyscip.SCIP_Bool   performpresol
#         pyscip.SCIP_Bool   branchedunbdvar
#         pyscip.SCIP_Bool   disableenforelaxmsg

# # cdef extern from "scip/type_history.h":
# #     ctypedef struct SCIP_HISTORY:
# #         pyscip.SCIP_Real   pscostcount [2]
# #         pyscip.SCIP_Real   pscostweightedmean [2]
# #         pyscip.SCIP_Real   pscostvariance [2]
# #         pyscip.SCIP_Real   vsids [2]
# #         pyscip.SCIP_Real   conflengthsum [2]
# #         pyscip.SCIP_Real   inferencesum [2]
# #         pyscip.SCIP_Real   cutoffsum [2]
# #         pyscip.SCIP_Longint    nactiveconflicts [2]
# #         pyscip.SCIP_Longint    nbranchings [2]
# #         pyscip.SCIP_Longint    branchdepthsum [2]

# # cdef extern from "time.h":
# #     ctypedef long clock_t
# # cdef extern from "scip/struct_clock.h":
# #     ctypedef struct SCIP_CPUCLOCK:
# #         pyscip.clock_t     user

# #     ctypedef struct SCIP_WALLCLOCK:
# #         long    sec
# #         long    usec

# #     ctypedef union SCIP_CLOCK_DATA:
# #         pyscip.SCIP_CPUCLOCK   cpuclock
# #         pyscip.SCIP_WALLCLOCK   wallclock

# #     # ctypedef enum SCIP_CLOCKTYPE:
# #     #     SCIP_CLOCKTYPE_DEFAULT = 0
# #     #     SCIP_CLOCKTYPE_CPU     = 1
# #     #     SCIP_CLOCKTYPE_WALL    = 2

# #     ctypedef struct SCIP_CLOCK:
# #         # SCIP_CLOCK_DATA  data
# #         pyscip.SCIP_Real        lasttime
# #         int              nruns
# #         pyscip.SCIP_CLOCKTYPE   clocktype
# #         pyscip.SCIP_Bool        usedefault
# #         pyscip.SCIP_Bool        enabled

# cdef extern from "scip/lp.h":
#     void SCIPlpRecalculateObjSqrNorm(SCIP_SET* set, SCIP_LP* lp)
#     pyscip.SCIP_Real SCIPcolGetObj(SCIP_COL *col)
#     pyscip.SCIP_Real SCIProwGetDualsol(SCIP_ROW* row)

# cdef extern from "scip/pub_lp.h":
#     # Row Methods
#     pyscip.SCIP_Real SCIProwGetLhs(SCIP_ROW* row)
#     pyscip.SCIP_Real SCIProwGetRhs(SCIP_ROW* row)
#     pyscip.SCIP_Real SCIProwGetConstant(SCIP_ROW* row)
#     int SCIProwGetLPPos(SCIP_ROW* row)
#     pyscip.SCIP_BASESTAT SCIProwGetBasisStatus(SCIP_ROW* row)
#     pyscip.SCIP_Bool SCIProwIsIntegral(SCIP_ROW* row)
#     pyscip.SCIP_Bool SCIProwIsLocal(SCIP_ROW* row)
#     pyscip.SCIP_Bool SCIProwIsModifiable(SCIP_ROW* row)
#     pyscip.SCIP_Bool SCIProwIsRemovable(SCIP_ROW* row)
#     int SCIProwGetNNonz(SCIP_ROW* row)
#     int SCIProwGetNLPNonz(SCIP_ROW* row)
#     SCIP_COL** SCIProwGetCols(SCIP_ROW* row)
#     pyscip.SCIP_Real* SCIProwGetVals(SCIP_ROW* row)
#     # Column Methods
#     int SCIPcolGetLPPos(SCIP_COL* col)
#     pyscip.SCIP_BASESTAT SCIPcolGetBasisStatus(SCIP_COL* col)
#     pyscip.SCIP_Bool SCIPcolIsIntegral(SCIP_COL* col)
#     SCIP_VAR* SCIPcolGetVar(SCIP_COL* col)
#     pyscip.SCIP_Real SCIPcolGetPrimsol(SCIP_COL* col)
#     pyscip.SCIP_Real SCIPcolGetLb(SCIP_COL* col)
#     pyscip.SCIP_Real SCIPcolGetUb(SCIP_COL* col)
#     int SCIPcolGetNLPNonz(SCIP_COL* col)
#     int SCIPcolGetNNonz(SCIP_COL* col)
#     SCIP_ROW** SCIPcolGetRows(SCIP_COL* col)
#     pyscip.SCIP_Real* SCIPcolGetVals(SCIP_COL* col)
#     int SCIPcolGetIndex(SCIP_COL* col)

# cdef extern from "execute.c":
#     int SCIPgetNLPCols(SCIP* scip)
#     int SCIPgetNLPRows(SCIP* scip)
# cdef extern from "execute.c":

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
