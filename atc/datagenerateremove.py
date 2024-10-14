import numpy as np
from backgroundfixed import generate_bg, plot_bg
from linear import LinearSystem
import stlpy.benchmarks.common as sbc
import stlpy.STL.predicate as ssp
import math
from  stlpy.solvers import GurobiMICPSolver as MICPSolver
from concurrent.futures import ProcessPoolExecutor


class SingleIntegrator(LinearSystem):
    def __init__(self, d, noise_scale):
        I = np.eye(d)
        z = np.zeros((d,d))
        # A = np.block([[I,z],
        #               [z,I]])
        A = np.block([I])
        B = np.block([I])
        C = np.block([I])
        D = np.block([z])
        # noise_aleatoric = noise_scale*np.random.randn(1)

        LinearSystem.__init__(self,A,B,C,D,noise_scale=noise_scale)

# print(a.dynamics_fcn)
def generate_atc(file_name):
    while 1:
        n = 4

        A = np.eye(n)
        B = np.eye(n)
        C = np.eye(n)
        D = np.eye(n)

        # print(b_g)
        sys = LinearSystem(A,B,C,D,noise_scale=0)

        b_g = generate_bg()

        OBSTACLE1 = b_g['obstacle_1']
        OBSTACLE2 = b_g['obstacle_2']
        OBSTACLE3 = b_g['obstacle_3']
        BACKGROUND = b_g['background']
        RUNWAY = b_g['runway_tuple']
        GOAL = b_g['goal_tuple']
        TRACK = b_g['track']

        # plot_bg(b_g=b_g)
        gamma_o11 = sbc.outside_rectangle_formula(OBSTACLE1,0,1,n)
        gamma_o12 = sbc.outside_rectangle_formula(OBSTACLE1,2,3,n)
        gamma_o21 = sbc.outside_rectangle_formula(OBSTACLE2,0,1,n)
        gamma_o22 = sbc.outside_rectangle_formula(OBSTACLE2,2,3,n)
        gamma_o31 = sbc.outside_rectangle_formula(OBSTACLE3,0,1,n)
        gamma_o32 = sbc.outside_rectangle_formula(OBSTACLE3,2,3,n)

        gamma_b1 = sbc.inside_rectangle_formula(BACKGROUND,0,1,n)
        gamma_b2 = sbc.inside_rectangle_formula(BACKGROUND,2,3,n)

        # gamma_r1 = sbc.inside_rectangle_formula(RUNWAY,0,1,n)
        # gamma_r2 = sbc.inside_rectangle_formula(RUNWAY,5,6,n)

        # gamma_t1 = sbc.inside_rectangle_formula(TRACK,0,1,n)
        # gamma_t2 = sbc.inside_rectangle_formula(TRACK,5,6,n)

        gamma_g1 = sbc.inside_rectangle_formula(GOAL,0,1,n)
        gamma_g2 = sbc.inside_rectangle_formula(GOAL,2,3,n)
        gamma_g2_t1 = sbc.outside_rectangle_formula(RUNWAY,2,3,n)

        var_phi_11 = gamma_g1.eventually(0,b_g['T_1'])&gamma_b1.always(0,b_g['T_2'])
        var_phi_21 = gamma_g2.eventually(0,b_g['T_2'])&gamma_b2.always(0,b_g['T_2'])

        var_phi_12 = gamma_o11.always(0, b_g['T_2']) & gamma_o21.always(0, b_g['T_2']) &\
                    gamma_o31.always(0, b_g['T_2'])
        var_phi_22 = gamma_o12.always(0, b_g['T_2']) & gamma_o22.always(0, b_g['T_2']) &\
                    gamma_o32.always(0, b_g['T_2'])

        d = ssp.LinearPredicate(a=[-1,-1,1,1],b=b_g['d_safe'])
        var_phi_3 = d.always(0,b_g['T_1'])

        var_phi_4 = gamma_g2_t1.always(0,b_g['T_1'])

        # var_phi_
        # var_phi = [var_phi_11,var_phi_21,var_phi_12,var_phi_22,var_phi_3,var_phi_4]
        var_phi = var_phi_11&var_phi_21&var_phi_12&var_phi_22&var_phi_3&var_phi_4
        Q = np.zeros([n,n])
        R = np.eye(n)
        x0 = np.array([b_g['flight_1'][0,0],b_g['flight_1'][1,0],b_g['flight_2'][0,0],b_g['flight_2'][1,0]])
        # print(x0)
        u_limit = 13
        u_limit_angle = math.pi/2
        u_min = np.array([-u_limit, -u_limit, -u_limit, -u_limit])
        u_max = np.array([u_limit, u_limit,  u_limit, u_limit])

        solver = MICPSolver(var_phi, sys, x0, b_g['T_2'], robustness_cost=True,M=1000,presolve=False)
        solver.AddControlBounds(u_min=u_min, u_max=u_max)
        solver.AddQuadraticCost(Q=Q,R=R)
        # solver.model.setParam('Cuts', 0)  # Disable cut generation
        # solver.model.setParam('Heuristics', 0.0)  # Turn off heuristics
        solver.model.setParam('RelaxLiftCuts', 0)
        res =solver.feasible_check()
        if res == 1:
            print('instance feasible')
            break
        else:
            print('instance not feasible')
    # file_name = './GCN_t/atcdata/train/test.mps'
    # for i in range(solver.model.SolCount):
    # solver.model.Params.SolutionNumber = 9
    # solver.model.write(file_name[:-4]+".sol")
    solver.Save(file_name=file_name)
    
# x, u, _, _ = solver.Solve()

# b_g = generate_bg()
for i in range(10):
    file_names = []
    for j in range(10):
        file_names.append('../smallatc/train/instances{}.mps'.format(10*i+j+1))
    print(file_names)
    with ProcessPoolExecutor() as executor:
        executor.map(generate_atc,file_names)

for i in range(10):
    file_names = []
    for j in range(10):
        file_names.append('../smallatc/test/instances{}.mps'.format(10*i+j+1))
    print(file_names)
    with ProcessPoolExecutor() as executor:
        executor.map(generate_atc,file_names)
# for i in range(1000):
#     file_name = './GCN_t/atcdata/train/instances{}.mps'.format(i+1)
#     generate_atc(file_name=file_name)
file_names = []
for i in range(10):
    file_names.append('../smallatc/validation/instances{}.mps'.format(i+1))

with ProcessPoolExecutor() as executor:
    executor.map(generate_atc,file_names)

