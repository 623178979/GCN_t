#include <stdio.h>
#include <assert.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #include <numpy/arrayobject.h>
#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scip/type_result.h"
#include "scip/type_branch.h"
#include "scip/branch.h"
#include "scip/scip_branch.h"
#include "scip/struct_scip.h"
#include "scip/struct_mem.h"
#include "scip/struct_primal.h"
#include "scip/struct_scip.h"
#include "scip/struct_set.h"
#include "scip/struct_var.h"
#include "scip/debug.h"
#include "scip/lp.h"
#include "scip/pub_message.h"
#include "scip/pub_var.h"
#include "scip/var.h"
#include "scip/scip_numerics.h"
#include "scip/set.h"
#include "scip/tree.h"
#include "scip/branch_relpscost.h"




// SCIP_RESULT executeBranchRule(SCIP* scip, const char* name, SCIP_Bool allowaddcons) {
//     SCIP_BRANCHRULE* branchrule;
//     SCIP_RESULT result;
//     SCIP_RETCODE retcode;

//     /* Initialize SCIP and load default plugins, if not already done outside this function */
//     // SCIP_CALL( SCIPcreate(&scip) );
//     // SCIP_CALL( SCIPincludeDefaultPlugins(scip) );

//     /* Find the branching rule by name */
//     branchrule = SCIPfindBranchrule(scip, name);
//     if (branchrule == NULL) {
//         printf("Error, branching rule not found!\n");
//         return SCIP_DIDNOTFIND;  // Using SCIP_PLUGINNOTFOUND as a representative error
//     } else {
//         // retcode = SCIPexecBranchruleLp(branchrule, scip, allowaddcons, &result);
//         // retcode = SCIP_DECL_BRANCHEXECLP((scip, branchrule, allowaddcons, &result));
//         // SCIP_CALL(SCIPbranchExecLP(scip->mem->probmem, scip->set, scip->stat, scip->transprob, scip->origprob,
//         //   scip->tree, scip->reopt, scip->lp, scip->sepastore, scip->branchcand, scip->eventqueue, scip->primal->cutoffbound,
//         //   TRUE, &result) );
//         SCIP_CALL(SCIP_DECL_BRANCHEXECLP(scip, branchrule, allowaddcons, result));
//         // if (retcode != SCIP_OKAY) {
//         //     printf("Error executing branching rule!\n");
//         //     return retcode;
//         // }
//         // branchrule.branchexeclp(scip, branchrule, allowaddcons, &result);
//         return result; // Directly returning SCIP_RESULT may need adjustments based on your application's error handling strategy
//     }
// }

SCIP_RESULT executeBranchRule(PyObject *capsule, const char* name, SCIP_Bool allowaddcons){
    SCIP_BRANCHRULE* branchrule;
    SCIP* scip;
    SCIP_RESULT result;
    SCIP_VAR** lpcands;
    SCIP_Real* lpcandssol;
    SCIP_Real* lpcandsfrac;
    int nlpcands;
    SCIP_Bool error = FALSE;


    // // SCIP_RETCODE retcode;
    scip = (SCIP*) PyCapsule_GetPointer(capsule, "scip");
    branchrule = SCIPfindBranchrule(scip, name);
    // int a;
    // a = 3;
    // return a;
    if (branchrule == NULL) {
        printf("Error, branching rule not found!\n");
        return SCIP_DIDNOTFIND;
        // return 6;
    } else {
        /* get the candidates */
        SCIP_CALL( SCIPgetLPBranchCands(scip, &lpcands, &lpcandssol, &lpcandsfrac, &nlpcands, NULL, &error) );
        if (error || nlpcands == 0)
        {
            // *result = ;
            return SCIP_DIDNOTRUN;
        }
        SCIP_CALL( SCIPexecRelpscostBranching(scip, lpcands, lpcandssol, lpcandsfrac, nlpcands, FALSE, &result) );
        // SCIP_CALL( SCIPbranchExecLP(scip->mem->probmem, scip->set, scip->stat, scip->transprob, scip->origprob,
        //   scip->tree, scip->reopt, scip->lp, scip->sepastore, scip->branchcand, scip->eventqueue, scip->primal->cutoffbound,
        //   TRUE, &result) );
        // SCIP_CALL(execRelpscost(scip, branchrule, branchcands, branchcandssol, branchcandsfrac, nbranchcands, executebranching, &result) );
        return result;
        // return  7;
    }
    return 0;
}

// PyObject* getState(PyObject *capsule, PyObject *prev_state) {
//     // PyObject *prev_state = NULL;
//     // PyObject *prev_state;
//     SCIP *scip;  // Assuming you have a way to retrieve this, perhaps stored in self
//     PyObject *update;

//     // if (!PyArg_ParseTuple(args, "|O", &prev_state)) {
//     //     return NULL; // Error if argument parsing fails
//     // }

//     // Assuming self is a custom type that includes SCIP*
//     scip = (SCIP*) PyCapsule_GetPointer(capsule, "scip");
//     if (prev_state != NULL){
//         update = prev_state;
//     }
//     // update = (prev_state != NULL);

//     // Get SCIP columns and number of columns
//     SCIP_COL** cols = SCIPgetLPCols(scip);
//     int ncols = SCIPgetNLPCols(scip);

//     // Array declarations
//     npy_intp col_dims[1] = {ncols};
//     PyObject *col_types = NULL;
//     PyObject *col_coefs = NULL;

//     // Initialize NumPy array API
//     import_array();

//     if (!update) {
//         col_types = PyArray_SimpleNew(1, col_dims, NPY_INT32);
//         col_coefs = PyArray_SimpleNew(1, col_dims, NPY_FLOAT32);
//         // Continue for other arrays...
//     } else {
//         // Extract arrays from prev_state if updating
//         col_types = PyDict_GetItemString(prev_state, "col_types");
//         col_coefs = PyDict_GetItemString(prev_state, "col_coefs");
//         // Increase ref counts as needed or ensure borrowed references are handled correctly
//         Py_XINCREF(col_types);
//         Py_XINCREF(col_coefs);
//         // Continue for other arrays...
//     }

//     // Example SCIP loop to populate data
//     for (int i = 0; i < ncols; i++) {
//         int col_type = SCIPvarGetType(SCIPcolGetVar(cols[i]));
//         double col_coef = SCIPcolGetObj(cols[i]);

//         if (!update) {
//             ((int *)PyArray_DATA((PyArrayObject *)col_types))[i] = col_type;
//             ((float *)PyArray_DATA((PyArrayObject *)col_coefs))[i] = col_coef;
//         }
//     }

//     // Create a Python dict to return
//     PyObject *result = PyDict_New();
//     PyDict_SetItemString(result, "col_types", col_types);
//     PyDict_SetItemString(result, "col_coefs", col_coefs);

//     // Clean up: DECREF new objects created
//     Py_XDECREF(col_types);
//     Py_XDECREF(col_coefs);

//     return result;
// }


// int rowTest(PyObject *capsule){
//     if(!Py_IsInitialized()){
//         Py_Initialize();
//     }
//     SCIP* scip;
//     // import_array();
//     scip = (SCIP*) PyCapsule_GetPointer(capsule, "scip");
//     int ncols = SCIPgetNLPCols(scip);
//     // SCIP_COL** cols = SCIPgetLPCols(scip);
//     int nrows = SCIPgetNLPRows(scip);
//     return nrows;
// }

