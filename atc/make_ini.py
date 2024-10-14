import pyscipopt as scip
import glob
from concurrent.futures import ProcessPoolExecutor
def make_ini(instance):
    m = scip.Model()
    m.readProblem('{}'.format(instance))
    m.setParam('limits/solutions', 1)
    m.setIntParam('timing/clocktype', 2)
    # m.setRealParam('limits/time', 0.2) 
    # m.setIntParam('presolving/maxrounds', 0)
    # m.setIntParam('propagating/maxrounds',0)
    # m.setIntParam('propagating/maxroundsroot',0)
    # m.setParam("separating/maxrounds", 0)
    # m.setParam("separating/maxroundsroot", 0)
    # m.setParam('numerics/feastol', 1e-06)
    m.setParam('parallel/maxnthreads', 10)
    m.setHeuristics(2)
    m.optimize()
    sols = m.getBestSol()
    m.writeSol(sols,filename=instance[:-4]+'.sol',write_zeros=True)
    return m.getStatus()
instances_valid = glob.glob('../atcdata/train/*.mps') + glob.glob('../atcdata/test/*.mps') + glob.glob('../atcdata/validation/*.mps') 
status = []
batch_size = 10
a = len(instances_valid)//batch_size
print(a)
for i in range(a):
    instances = instances_valid[i*batch_size:i*batch_size+batch_size]
    with ProcessPoolExecutor() as executor:
        status = list(executor.map(make_ini,instances))
    print(status)
# for instance in instances_valid:
#     m = scip.Model()
#     m.readProblem('{}'.format(instance))
#     m.setParam('limits/solutions', 1)
#     # m.setIntParam('presolving/maxrounds', 0)
#     # m.setIntParam('propagating/maxrounds',0)
#     # m.setIntParam('propagating/maxroundsroot',0)
#     m.setParam('numerics/feastol', 1e-03)
#     m.setHeuristics(1)
#     m.optimize()
#     sols = m.getBestSol()
#     m.writeSol(sols,filename=instance[:-4]+'_scip.sol',write_zeros=True)