# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:03:25 2026

@author: Jules
"""


import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from opty.utils import parse_free

from generate_eom_model_level_0 import generate_model

from utils import plot_trajectories

t, x, r, eoms, p= generate_model()

NUM_STATES = len(x)
NUM_INPUTS = 1
NUM_NODES = 10
WEIGHT = 0.95
SPEED = 1.0  # m/s
DURATION = 5
NUM_NODES = 200
INTERVAL_VALUE = DURATION / (NUM_NODES - 1)




def obj(free):
    """Minimize the sum of the squares of steer torque"""
    x, r, _ = parse_free(free, NUM_STATES, NUM_INPUTS, NUM_NODES)
    q3 = x[2]
    
    #Minimize headind
    
    return INTERVAL_VALUE*(WEIGHT*np.sum(q3**2) + (1.0-WEIGHT)*np.sum(r.flatten())**2)


def obj_grad(free):
    x, r, _ = parse_free(free, NUM_STATES, NUM_INPUTS, NUM_NODES)
    q3 = x[2]
    grad = np.zeros_like(free)
    
    grad[2*NUM_NODES:3*NUM_NODES] = 2.0*INTERVAL_VALUE*WEIGHT*q3
    grad[NUM_STATES*NUM_NODES:(NUM_STATES + NUM_INPUTS)*NUM_NODES] = 2.0*(1.0-WEIGHT)*INTERVAL_VALUE*r.flatten()
    return grad

q1, q2, q3, q4, q6, q7, q8, q5 = x[:8]
u4, u6, u7, u1, u2, u3, u5, u8 = x[8:]

bounds = {
    q1: (-0.1, SPEED*DURATION + 0.1),
    q2: (-0.1, SPEED*DURATION*0.25),
    q3: (-1.0, 1.0),
    q4: (-1.0, 1.0),
    q5: (-1.0, 1.0),
    q6: (-100.0, 100.0),
    q7: (-1.0, 1.0),
    q8: (-100.0, 100.0),
    u1: (0.0, 10.0),
    u2: (-5.0, 5.0),
    u3: (-2.0, 2.0),
    u4: (-4.0, 4.0),
    u5: (-4.0, 4.0),
    u6: (-20.0, 0.0),
    u7: (-4.0, 4.0),
    u8: (-20.0, 0.0),
}

problem = Problem(
    obj,
    obj_grad,
    eoms,
    x,
    NUM_NODES,
    INTERVAL_VALUE,
    known_parameter_map=p,
    # instance_constraints=instance_constraints,
    bounds=bounds,
    #integration_method='midpoint',
    parallel=True,
)

# problem.add_option('nlp_scaling_method', 'gradient-based')
# problem.add_option('linear_solver', 'spral')

initial_guess = np.zeros((NUM_STATES+NUM_INPUTS)*NUM_NODES)

# Find the optimal solution.
sol, info = problem.solve(initial_guess)