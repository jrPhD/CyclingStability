"""
Created on Tue Nov 25 11:08:14 2025

@author: Jules + EveCharbie
"""

# =============================================================================
# The Default Bicycle Model
# From https://mechmotum.github.io/symbrim/tutorials/my_first_bicycle.html
# Let's start with a very basic non-linear bicycle without rider
# =============================================================================


# Requirements
# pip install symbrim
# pip install bicycleparameters
# pip install symmeplot
# conda install -c conda-forge casadi

import pickle
import platform
import warnings
from IPython.display import display
import numpy as np
#import casadi as cas

import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics.mechanics import System

import symbrim as sb
from symbrim.bicycle import RigidRearFrameMoore, WhippleBicycleMoore
from symbrim.brim import SideLeanSeatSpringDamper
from symbrim.rider import PinElbowTorque, SphericalShoulderTorque
from sympy.utilities.lambdify import lambdify

from typing import Dict, Any

def generate_continuous_random_motion(num_terms=12):
    """
    Generates a continuous, random-like function of time 't'.
    Returns both the symbolic expression and a numerical function for evaluation.
    """
    t = sm.symbols('t')
    
    np.random.seed() 
    
    terms = []
    for i in range(num_terms):

        amplitude = np.random.uniform(0.1, 0.5)
        frequency = np.random.uniform(0.5, 15.0)
        phase = np.random.uniform(0, 2 * np.pi)
        
        term = amplitude * sm.sin(frequency * t + phase)
        terms.append(term)
    
    expression = sum(terms)
    
    # Create a numerical function for fast evaluation with numpy arrays
    func_numeric = sm.lambdify(t, expression, modules=['numpy'])
    
    return expression, func_numeric

def generate_model():
    t = me.dynamicsymbols._t
    bicycle = sb.WhippleBicycle("bike_v1_0")
    assert type(bicycle) is WhippleBicycleMoore
    bicycle.rear_frame = sb.RigidRearFrame.from_convention("moore", "rear_frame")
    assert type(bicycle.rear_frame) is RigidRearFrameMoore
    bicycle.rear_wheel = sb.KnifeEdgeWheel("rear_wheel")
    bicycle.rear_tire = sb.NonHolonomicTire("rear_tire")
    
    bicycle.ground = sb.FlatGround("ground")
    bicycle.front_frame = sb.RigidFrontFrame("front_frame")
    bicycle.front_wheel = sb.KnifeEdgeWheel("front_wheel")
    bicycle.front_tire = sb.NonHolonomicTire("front_tire")
    
    assert len(bicycle.submodels) == 5
    assert len(bicycle.connections) == 2
    
    bicycle.define_all()
    system = bicycle.to_system()
    
    normal = bicycle.ground.get_normal(bicycle.ground.origin)
    
    # Add loads and actuators
    
    # Gravity
    g = sm.symbols("g")
    system.apply_uniform_gravity(-g * normal)
    
    # Disturbance
    # disturbance = me.dynamicsymbols("disturbance")
    # system.add_loads(me.Force(bicycle.rear_frame.saddle.point, disturbance * bicycle.rear_frame.wheel_hub.axis))
    
    # Steer torque
    steer_torque = me.dynamicsymbols("steer_torque")
    system.add_actuators(
        me.TorqueActuator(
            steer_torque,
            bicycle.rear_frame.steer_hub.axis,
            bicycle.rear_frame.steer_hub.frame,
            bicycle.front_frame.steer_hub.frame,
        )
    )
    
    # Before forming the EoMs we need to specify which generalized coordinates
    # and speeds are independent and which are dependent.
    
    #q indep : q1, q2, q3, q4, q6, q7, q8
    #u indep : u4, u6, u7
    
    system.q_ind = [*bicycle.q[:4], *bicycle.q[5:]]
    system.q_dep = [bicycle.q[4]]
    system.u_ind = [bicycle.u[3], *bicycle.u[5:7]]
    system.u_dep = [*bicycle.u[:3], bicycle.u[4], bicycle.u[7]]
    system.validate_system()
    
    try:
        system.validate_system()
    except ValueError as e:
        print("\n\nERROR : ")
        display(e)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eoms = system.form_eoms(constraint_solver="CRAMER")
    
    # The equations of motions are generated as a
    #                                   "sympy.matrices.dense.MutableDenseMatrix"
    
    # %% Parametrization
    
    import bicycleparameters as bp
    
    bike_params = bp.Bicycle("Browser", pathToData="data")
    # bike_params.add_rider("Jason", reCalc=True)
    
    constants = bicycle.get_param_values(bike_params)
    constants[g] = 9.81  # Don't forget to specify the gravitational constant.
    
    print("\n\nConstants of the model:")
    print(constants)
    
    missing_symbols = bicycle.get_all_symbols().difference(constants.keys())
    
    print("\n\nIs there any missing constant? -->")
    print(missing_symbols)
    
    # dist = sm.sin(2*t*3.14*10)
    
    # eoms = eoms.col_join(sm.Matrix([disturbance - dist]))
    
    x = system.q.col_join(system.u)
    r = steer_torque
    p = constants

    return t, x, r, eoms, p


def export_constants(constants: dict[str, float]) -> None:
    """
    Export the constants to a pickle file for later use.

    Parameters
    ----------
    constants : Dictionary of {symbol: value} for constant parameters
    """

    if platform.system() == "Windows":
        full_file_name = f"model_files\constants_d.pkl"
    else:
        full_file_name = f"model_files/constants_d.pkl"

    with open(full_file_name, "wb") as f:
        pickle.dump(constants, f)






if __name__ == "__main__":

    pass

    t, x, r, eoms, p= generate_model()

    # export_constants(constants)



