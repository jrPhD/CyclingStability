# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:03:25 2026

@author: Jules
"""


import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty import Problem
from opty.utils import parse_free

from generate_eom_model_level_0 import generate_model

from utils import plot_trajectories


import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

from symbrim.utilities.plotting import Plotter

model_names = ['m1','m2']

t1, x1, r1, eoms1, p1, bicycle1, disturbance_1 = generate_model(model_names[0])
t2, x2, r2, eoms2, p2, bicycle2, disturbance_2 = generate_model(model_names[1])

t = t1
x = x1.col_join(x2)
r = r1.col_join(r2)
eoms = eoms1.col_join(eoms2)
bicycles = [bicycle1, bicycle2]
p = p1
disturbance = disturbance_1.col_join(disturbance_2)

param, param_vals = zip(*p.items())




NUM_MODELS = 2
NUM_STATES = len(x1)
NUM_INPUTS = len(r1)

NUM_STATES_TOT = NUM_MODELS*NUM_STATES
NUM_INPUTS_TOT = NUM_MODELS*NUM_INPUTS

WEIGHT = 0.5
SPEED = 5  # m/s
DURATION = 1.5
NUM_NODES = 10
INTERVAL_VALUE = DURATION / (NUM_NODES - 1)



def animate_solution(t_simu, x_opt, T_opt, disturbance, bicycle, x, r):

    # Create some functions to interpolate the results.
    x_eval = CubicSpline(t_simu, x_opt.T)
    r_eval = CubicSpline(t_simu, T_opt.T)
    # dis_eval = CubicSpline(t_simu, dis.T)
    # max_disturbance = dis.max()

    # Plot the initial configuration of the model
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))
    plotter = Plotter.from_model(bicycle, ax=ax)
    # plotter.add_vector(
    #     disturbance * bicycle.rear_frame.wheel_hub.axis / max_disturbance,
    #     bicycle.rear_frame.saddle.point,
    #     name="disturbance",
    #     color="r",
    # )
    # plotter.lambdify_system((x, r, disturbance, param))
    # plotter.evaluate_system(x_eval(0.0), r_eval(0.0),dis[0], param_vals)
    
    plotter.lambdify_system((x, r, param))
    plotter.evaluate_system(x_eval(0.0), r_eval(0.0), param_vals)
    plotter.plot()
    X, Y = np.meshgrid(np.arange(-1, 10, 0.5), np.arange(-1, 3, 0.5))
    ax.plot_wireframe(X, Y, np.zeros_like(X), color="k", alpha=0.3, rstride=1, cstride=1)
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.view_init(19, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    fps = 30
    # ani = plotter.animate(
    #     lambda ti: (x_eval(ti), r_eval(ti), dis_eval(ti), param_vals), frames=np.arange(0, t_simu[-1], 1 / fps), blit=False
    # )
    
    ani = plotter.animate(
        lambda ti: (x_eval(ti), r_eval(ti), param_vals), frames=np.arange(0, t_simu[-1], 1 / fps), blit=False
    )
    
    display(HTML(ani.to_jshtml(fps=fps)))

def obj(free):
    """Minimize the sum of the squares of steer torque"""
    x, r, _ = parse_free(free, NUM_STATES, NUM_INPUTS, NUM_NODES)
    q3 = x[2]
    # u1 = x[8]
    # q1 = x[0]
    q4 = x[3]
    
    J1 = INTERVAL_VALUE*((WEIGHT)*(np.sum(q4)**2))
    J2 = INTERVAL_VALUE*((WEIGHT)*(np.sum(q3)**2))

    J3 = INTERVAL_VALUE*((1-WEIGHT)*(np.sum((r.flatten())**2)))
    
    #Minimize headind
    
    return J1+J2+J3


def obj_grad(free):
    x, r, _ = parse_free(free, NUM_STATES, NUM_INPUTS, NUM_NODES)
    q3 = x[2]
    # u1 = x[8]
    # q1 = x[0]
    q4 = x[3]



    grad = np.zeros_like(free)
    
    grad[3*NUM_NODES:4*NUM_NODES] = 2.0*INTERVAL_VALUE*WEIGHT*q4
    grad[2*NUM_NODES:3*NUM_NODES] = 2.0*INTERVAL_VALUE*WEIGHT*q3
    
    # grad[0*NUM_NODES:1*NUM_NODES] = 2.0*INTERVAL_VALUE*WEIGHT*q1

    grad[NUM_STATES*NUM_NODES:(NUM_STATES + NUM_INPUTS)*NUM_NODES] = 2.0*(1.0-WEIGHT)*INTERVAL_VALUE*r.flatten()
    return grad


def generate_bounds_and_contstraints():
    
    bounds = {}
    initial_state_constraints = {}
    final_state_constraints = {}
    
    for model_index in range(NUM_MODELS):

        q1, q2, q3, q4, q5, q6, q7, q8,  = x[model_index*8:8*(model_index+1)]
        u1, u2, u3, u4, u5, u6, u7, u8 = x[(model_index+1)*8:(model_index+2)*8]
        steer_torque = r[model_index]

        bounds[q1]= (-0.1, SPEED*DURATION + 0.1)
        bounds[q2]= (-3, 3) #Si large la vélo est tabilisé mais part sur le coté
        bounds[q3]= (-1.0, 1.0)
        bounds[q4]= (-1.0, 1.0)
        bounds[q5]= (-1, 1)
        bounds[q6]= (-100.0, 100.0)
        bounds[q7]= (-1.0, 1.0)
        bounds[q8]= (-100.0, 100.0)
        bounds[u1]= (0, 10) #ok
        bounds[u2]= (-5, 5) #ok
        bounds[u3]= (-2.0, 2.0)
        bounds[u4]= (-4.0, 4.0)
        bounds[u5]= (-4.0, 4.0)
        bounds[u6]= (-20.0, 0.0)
        bounds[u7]= (-4.0, 4.0) #ok
        bounds[u8]= (-20.0, 0.0)
        # pedaling_torque: (-100,100),
        bounds[steer_torque] = (-25,25) #ok


        initial_state_constraints[q1] = 0.0
        initial_state_constraints[q2] = 0.0
        initial_state_constraints[q3] = 0.0
        # initial_state_constraints[q4] = 0.0
        # initial_state_constraints[q5] = 0.4
        initial_state_constraints[q6] = 0.0
        initial_state_constraints[q7] = 0.0
        initial_state_constraints[q8] = 0.0
        initial_state_constraints[u1] = SPEED
        initial_state_constraints[u2] = 0.0
        initial_state_constraints[u3] = 0.0
        initial_state_constraints[u4] = 0.0
        # initial_state_constraints[u5] = 0.0
        initial_state_constraints[u6] = -SPEED/param_vals[-6]
        initial_state_constraints[u7] = 0.0
        initial_state_constraints[u8] = -SPEED/param_vals[-2]
    


        # final_state_constraints[q1] = SPEED*DURATION
        # final_state_constraints[q2] = 0.0
        # final_state_constraints[q3] = 0.0
        # final_state_constraints[q4] = 0.0
        # final_state_constraints[q5] = -0.314
        # final_state_constraints[q6] = 0.0
        # final_state_constraints[q7] = 0.0
        # final_state_constraints[q8] = 0.0
        # final_state_constraints[u1] = SPEED
        # final_state_constraints[u2] = 0.0
        final_state_constraints[u3] = 0.0
        final_state_constraints[u4] = 0.0
        # final_state_constraints[u5] = 0.0
        # final_state_constraints[u6] = -SPEED/param_vals[-6]
        final_state_constraints[u7] = 0.0
        # final_state_constraints[u8] = -SPEED/param_vals[-2]
        
    return(bounds, initial_state_constraints, final_state_constraints)


bounds, initial_state_constraints, final_state_constraints = generate_bounds_and_contstraints()


instance_constraints = tuple(
        xi.replace(t, 0.0) - xi_val for xi, xi_val in initial_state_constraints.items()) + tuple(
        xi.replace(t, DURATION) - xi_val for xi, xi_val in final_state_constraints.items())

# disturbance = me.dynamicsymbols("disturbance")
# dis_1 = np.random.normal(0, 100, NUM_NODES)
# dis_2 = np.random.normal(0, 100, NUM_NODES)



known_trajectories = {dist : np.random.normal(0, 20, NUM_NODES) for dist in disturbance}


problem = Problem(
    obj,
    obj_grad,
    eoms,
    x,
    NUM_NODES,
    INTERVAL_VALUE,
    known_parameter_map = p,
    known_trajectory_map = known_trajectories,
    instance_constraints=instance_constraints,
    bounds=bounds,
    # integration_method='midpoint',
    time_symbol=t,
    parallel=True,
    # backend='numpy'
)
problem.add_option('max_iter' , 100000)



# initial_guess = np.random.rand((NUM_STATES+NUM_INPUTS)*NUM_NODES)
initial_guess = np.zeros((NUM_STATES_TOT+NUM_INPUTS_TOT)*NUM_NODES)


# Find the optimal solution.
sol, info = problem.solve(initial_guess)

problem.plot_objective_value()
problem.plot_constraint_violations(sol)
problem.plot_trajectories(sol)

visualization_flag = True
x_opt, T_opt = sol.reshape(-1, NUM_NODES)[:-1,:], sol.reshape(-1, NUM_NODES)[-1:,:]
t_simu = np.linspace(0, DURATION, NUM_NODES)

if visualization_flag:
    
    for model_index in range(NUM_MODELS):
        
        x_opt_model = x_opt[model_index*16:16*(model_index+1)]
        T_opt_model = T_opt[model_index*16:16*(model_index+1)]
        disturbance_model =  known_trajectories[list(known_trajectories.keys())[model_index]]
        x_model = x[model_index*16:16*(model_index+1)]
        r_model = r[model_index*NUM_INPUTS:NUM_INPUTS*(model_index+1)][0]
    
    
        animate_solution(t_simu, 
                         x_opt_model, 
                         T_opt_model, 
                         disturbance_model, 
                         bicycles[model_index], 
                         x_model, 
                         r_model)





