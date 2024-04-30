#include <stdio.h>
#include <assert.h>
#include <Python.h>
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
// int executeBranchRule(PyObject *capsule, const char* name, SCIP_Bool allowaddcons){
//     return 7;
// }

// SCIP_BRANCHRULE* executeBranchRule(SCIP* scip, const char* name){
//     SCIP_BRANCHRULE* branchrule;
//     // SCIP_RESULT result;
//     // SCIP_RETCODE retcode;

//     branchrule = SCIPfindBranchrule(scip, name);
//     // if (branchrule == NULL) {
//     //     printf("Error, branching rule not found!\n");
//     //     return SCIP_DIDNOTFIND;
//     // } else {
//     //     SCIP_CALL( branchrule -> branchexeclp(scip, branchrule, allowaddcons, result) );
//     //     return result;
//     // }
//     return branchrule;
// }
// int main() {
//     SCIP* scip = NULL;
//     const char* branchrule_name = "relpscost";
//     SCIP_Bool allowaddcons = TRUE;
//     SCIP_RESULT result;

//     /* Call the function */
//     result = executeBranchRule(scip, branchrule_name, allowaddcons);
//     printf("Branch rule execution result: %d\n", result);

//     /* Free SCIP resources, assuming SCIP was initialized in executeBranchRule */
//     SCIP_CALL( SCIPfree(&scip) );

//     return 0;
// }
// SCIP_BRANCHRULE* main(){

//     // return 0;
// }
