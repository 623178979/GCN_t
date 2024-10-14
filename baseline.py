import pyscipopt as scip
import glob
import time
import os
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor

instances_train = []
instances_train += ['./smallatc/test/instances{}.mps'.format(i+1) for i in range(50)]
result = []
timelimit_list = [10,20,30,40,50,60,70,80,90,100,200,500,1000]
def solve(instance):
    # print(instance)
    m = scip.Model()
    # m.setIntParam('display/verblevel', 0)
    m.readProblem('{}'.format(instance))
    m.setIntParam('timing/clocktype', 2)
    m.setParam('limits/time',timelimit)
    m.readSol(instance[:-4]+'.sol')
    # current setting is for default scip, to get the result of no-presolving scip, 
    # uncomment following 4 lines

    # m.setIntParam('presolving/maxrounds', 0)
    # m.setIntParam('presolving/maxrestarts', 0)
    # m.setParam("separating/maxrounds", 0)
    # m.setParam("separating/maxroundsroot", 0)
    m.optimize()
    res = m.getPrimalbound()
    return res



for timelimit in timelimit_list:
    filename = 'samescip801_baseline'+str(timelimit)+'s.csv'
    os.makedirs('./scip_baselines',exist_ok=True)
    fieldnames = [
        'instance',
        'object_value',
        'mean',
    ]
    
    with open('./scip_baselines/{}'.format(filename),'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # for i in range(5):
        #     batch = instances_train[i*10:i*10+10]
        with ProcessPoolExecutor() as executor:
            result = list(executor.map(solve,instances_train))
            # for i in range(len(instances_train)):
            #     instance = instances_train[i]
            #     print(instance)
            #     m = scip.Model()
            #     m.setIntParam('display/verblevel', 0)
            #     m.readProblem('{}'.format(instance))
            #     m.setParam('limits/time',timelimit)
            #     # a = time.time()
            #     m.optimize()
            #     res = m.getObjVal()
            #     print('getObjVal:',res)

            #     result.append(res)
        for j in range(len(instances_train)):
            writer.writerow({
                'instance':instances_train[j],
                'object_value':result[j],
                'mean':np.nanmean(result)
            })
            csvfile.flush()
