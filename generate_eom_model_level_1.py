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
import bicycleparameters as bp


# model_name = 'model_1'

def generate_model(model_name):
    t = me.dynamicsymbols._t
    bicycle = sb.WhippleBicycle(f"{model_name}")
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
    
    
    #Noise
    
    w_s_q4 = me.dynamicsymbols(f"{model_name}_w_s_q4")
    w_s_u4 = me.dynamicsymbols(f"{model_name}_w_s_u4")
    w_s_q7 = me.dynamicsymbols(f"{model_name}_w_s_q7")
    w_s_u7 = me.dynamicsymbols(f"{model_name}_w_s_u7")
    
    w_m_steer = me.dynamicsymbols(f"{model_name}_w_m_steer")
    w_m_roll = me.dynamicsymbols(f"{model_name}_w_m_roll")
    
    W_s = sm.Matrix([w_s_q4, w_s_u4, w_s_q7, w_s_u7])
    W_m = sm.Matrix([w_m_steer, w_m_roll])

    W = W_s.col_join(W_m)
    
    # Disturbance
    disturbance = me.dynamicsymbols(f"{model_name}_disturbance")
    # disturbance = f_disturb(t)
    system.add_loads(me.Force(bicycle.rear_frame.saddle.point, disturbance * bicycle.rear_frame.wheel_hub.axis))
    
    # Steer torque
    steer_torque_vol = me.dynamicsymbols("steer_torque_vol")


    steer_torque = me.dynamicsymbols("steer_torque")
    system.add_actuators(
        me.TorqueActuator(
            steer_torque,
            bicycle.rear_frame.steer_hub.axis,
            bicycle.rear_frame.steer_hub.frame,
            bicycle.front_frame.steer_hub.frame,
        )
    )
    
    # Roll torque
    # roll_torque = me.dynamicsymbols(f"{model_name}_roll_torque")
    roll_torque = me.dynamicsymbols("roll_torque")
    roll_torque_vol = me.dynamicsymbols("roll_torque_vol")

    
    system.add_actuators(
        me.TorqueActuator(
            roll_torque,
            bicycle.rear_frame.body.frame.x,
            bicycle.ground.frame,
            bicycle.rear_frame.body.frame,
        )
    )
    
    
    # pedaling_torque = me.dynamicsymbols(f"{model_name}_pedaling_torque")
    pedaling_torque = me.dynamicsymbols("pedaling_torque")
    
    system.add_actuators(
        me.TorqueActuator(
            pedaling_torque,
            bicycle.rear_frame.wheel_hub.axis,
            bicycle.rear_wheel.frame,
            bicycle.rear_frame.wheel_hub.frame,
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
    
    
    bike_params = bp.Bicycle("Browser", pathToData="data")
    # bike_params.add_rider("Jason", reCalc=True)
    
    constants = bicycle.get_param_values(bike_params)
    constants[g] = 9.81  # Don't forget to specify the gravitational constant.
    
    # print("\n\nConstants of the model:")
    # print(constants)
    
    # missing_symbols = bicycle.get_all_symbols().difference(constants.keys())
    
    # print("\n\nIs there any missing constant? -->")
    # print(missing_symbols)
    
    
    # eoms = eoms.col_join(sm.Matrix([disturbance - f_disturb(t)]))
    
    x = system.q.col_join(system.u)
    # x = x.col_join(sm.Matrix([disturbance]))
    # r = (steer_torque, pedaling_torque)
    
    
    p = constants
    
    permutation = [0, 1, 2, 3, 7, 4, 5, 6, 11, 12, 13, 8, 14, 9, 10, 15]
    
    # x = x[[0,1,2,3,7,4,5,6,11,12,13,8,14,9,10,15]]
    x_reordered = sm.Matrix([x.row(i) for i in permutation])
    x = x_reordered.as_immutable()
    
    q1, q2, q3, q4, q5, q6, q7, q8,  = x[:8]
    u1, u2, u3, u4, u5, u6, u7, u8 = x[8:]
    
    nh_cons = system.nonholonomic_constraints
    h_cons = system.holonomic_constraints
    kdes = system.kdes
    
    #Feedback loops
    
    ## Option 1: feedback gain is constant over time
    K_s_q4 = sm.symbols("K_s_q4")
    K_s_q7 = sm.symbols("K_s_q7")
    K_s_u4 = sm.symbols("K_s_u4")
    K_s_u7 = sm.symbols("K_s_u7")
    
    
    K_r_q4 = sm.symbols("K_r_q4")
    K_r_q7 = sm.symbols("K_r_q7")
    K_r_u4 = sm.symbols("K_r_u4")
    K_r_u7 = sm.symbols("K_r_u7")
    
    M_gains_steer = sm.Matrix([K_s_q4, K_s_q7, K_s_u4, K_s_u7]).T
    M_gains_roll = sm.Matrix([K_r_q4, K_r_q7, K_r_u4, K_r_u7]).T
    
    
    q_obs = sm.Matrix([q4, q7, u4, u7])
    q_ref = sm.Matrix([0, 0, 0, 0])
    
    eq_feedback_1 = sm.Matrix([steer_torque - steer_torque_vol + w_m_steer]) - M_gains_steer*(q_obs - q_ref + W_s)
    eq_feedback_2 = sm.Matrix([roll_torque - roll_torque_vol + w_m_roll]) - M_gains_roll*(q_obs - q_ref + W_s)
    
    
    k = sm.Matrix([K_s_q4, K_s_q7, K_s_u4, K_s_u7, K_r_q4, K_r_q7, K_r_u4, K_r_u7])
    
    r_dep = sm.Matrix([steer_torque, roll_torque])
    
    r_ind = sm.Matrix([pedaling_torque, steer_torque_vol, roll_torque_vol])
    
    
    
    
    
    eoms = eoms.col_join(sm.Matrix(kdes)).col_join(sm.Matrix(h_cons)).col_join(sm.Matrix(nh_cons))
    
    #Add feedback equations
    
    eoms = eoms.col_join(sm.Matrix([eq_feedback_1])).col_join(sm.Matrix([eq_feedback_2]))
    
    disturbance = sm.Matrix([disturbance])

    return t, x, r_dep, r_ind, k, eoms, p, bicycle, disturbance, W


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
    model_name = 'model_1'
    t, x, r_dep, r_ind, k, eoms, p, bicycle, disturbance, W = generate_model(model_name)

    # export_constants(constants)



