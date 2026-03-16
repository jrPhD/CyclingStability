# -*- coding: utf-8 -*-
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

from sympy_to_casadi import generate_model_file

from typing import Dict, Any


def create_symbrim_model(simulation_flag: bool = False, visualization_flag: bool = False):
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
    disturbance = me.dynamicsymbols("disturbance")
    system.add_loads(me.Force(bicycle.rear_frame.saddle.point, disturbance * bicycle.rear_frame.wheel_hub.axis))

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

    # %% Simulation

    if simulation_flag:
        from simulator import Simulator

        simu = Simulator(system)

        simu.constants = constants
        simu.initial_conditions = {
            **{xi: 0.0 for xi in system.q.col_join(system.u)},
            bicycle.q[4]: 0.314,  # Initial guess rear frame pitch.
            bicycle.u[5]: -3.0 / constants[bicycle.rear_wheel.radius],  # Rear wheel angular velocity.
        }
        roll_rate_idx = len(system.q) + system.u[:].index(bicycle.u[3])
        max_roll_rate, max_torque = 0.2, 10
        simu.inputs = {
            disturbance: lambda t, x: (30 + 30 * t) * np.sin(t * 2 * np.pi),
            steer_torque: lambda t, x: -max_torque * max(-1, min(x[roll_rate_idx] / max_roll_rate, 1)),
        }

        simu.initialize()
        print("Initial Conditions:")
        simu.initial_conditions

        simu.solve([0, 5], solver="solve_ivp")

    # %% Visualization

    if visualization_flag:

        import matplotlib.pyplot as plt
        from IPython.display import HTML
        from matplotlib.animation import FuncAnimation
        from scipy.interpolate import CubicSpline

        from symbrim.utilities.plotting import Plotter

        # Create some functions to interpolate the results.
        x_eval = CubicSpline(simu.t, simu.x.T)
        r_eval = CubicSpline(simu.t, [[cf(t, x) for cf in simu.inputs.values()] for t, x in zip(simu.t, simu.x.T)])
        p, p_vals = zip(*simu.constants.items())
        max_disturbance = r_eval(simu.t)[:, tuple(simu.inputs.keys()).index(disturbance)].max()

        # Plot the initial configuration of the model
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))
        plotter = Plotter.from_model(bicycle, ax=ax)
        plotter.add_vector(
            disturbance * bicycle.rear_frame.wheel_hub.axis / max_disturbance,
            bicycle.rear_frame.saddle.point,
            name="disturbance",
            color="r",
        )
        plotter.lambdify_system((system.q[:] + system.u[:], simu.inputs.keys(), p))
        plotter.evaluate_system(x_eval(0.0), r_eval(0.0), p_vals)
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
        ani = plotter.animate(
            lambda ti: (x_eval(ti), r_eval(ti), p_vals), frames=np.arange(0, simu.t[-1], 1 / fps), blit=False
        )
        display(HTML(ani.to_jshtml(fps=fps)))

    return system, constants, t


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


def eval_num_full(
    system: me.System, constants: dict[str, float], x: np.ndarray, tau: float, distu: float
) -> np.ndarray:
    """
    Evaluate the forward dynamics RHS numerically.

    Solves: M_m @ Xd + F_m = 0  =>  Xd = -M_m^{-1} @ F_m

    Parameters
    ----------
    system : The system object after form_eoms() has been called
    constants : Dictionary of {symbol: value} for constant parameters
    x : State vector [q, u]
    tau : Steer torque value
    distu : Disturbance value

    Returns
    -------
    xdot : Time derivative of the state vector
    """
    # Get the mass matrix and forcing vector from system
    M_m = system.mass_matrix_full
    F_m = system.forcing_full

    # Extract parameters and their values
    _p, _p_vals = zip(*constants.items())

    # Build state vector (q joined with u)
    _x = system.q.col_join(system.u)

    # Define input symbols
    steer_torque = me.dynamicsymbols("steer_torque")
    disturbance = me.dynamicsymbols("disturbance")

    # Use tolist() to convert matrix to nested list
    M_list = M_m.tolist()
    F_list = F_m.tolist()

    # Lambdify the nested lists
    f_M_m = lambdify((_x, _p, steer_torque, disturbance), M_list, modules="numpy", cse=True)

    f_F_m = lambdify((_x, _p, steer_torque, disturbance), F_list, modules="numpy", cse=True)

    # Evaluate numerically - convert to numpy arrays
    M_m_num = np.array(f_M_m(x, _p_vals, tau, distu), dtype=float)
    F_m_num = np.array(f_F_m(x, _p_vals, tau, distu), dtype=float).flatten()

    # Solve M_m @ xdot = F_m
    xdot = np.linalg.solve(M_m_num, F_m_num)

    print(f"Symbrim RHS : {xdot}")

    return xdot


# %% Conversion
def generate_casadi_file_full(system, constants): #Not used so far

    """Generates python file with casadi structure containing the full dynamics equations"""
    # Matrices extraction

    # M_m @ X_m + F_m = 0
    M_m = system.mass_matrix_full
    F_m = system.forcing_full

    variable_list = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "u1",
        "u2",
        "u3",
        "u4",
        "u5",
        "u6",
        "u7",
        "u8",
        "steer_torque",
        "disturbance",
    ]

    generate_model_file("model_d", ["M_m", "F_m"], [M_m, F_m], variable_list, constants)

    return M_m, F_m

def generate_casadi_file_indep_dynamics(system : System, constants : Dict[str, Any]) -> None:

    """Generates python file with casadi structure containing :
        - dynamics of indep generalized speed (u4, u6, u7),
        - 4 non-holomonic constrains,
        - 1 holonomic constrain,
        - 8 kinematics differential equations"""

    # 3 dynamics equations of indep variables, M_d @ X_d + F_d = 0
    M_u4_u6_u7 = system.mass_matrix[:3,:]
    F_u4_u6_u7 = system.forcing[:3,:]
    
    
    
    # 4 non-holonomic contstains + 1 holonomic constrain
    nh_cons = system.nonholonomic_constraints
    h_cons = system.holonomic_constraints
    
    vel_cons = system.velocity_constraints
    
    #kinematics differential equations u=q_dot
    kdes = system.kdes
    
    q_d = system.q.diff(t)
    # u_dep_d = system.u_dep.diff(t)
    # u_indep_d = system.u_ind.diff(t)
    # u_d = system.u_dep.col_join(system.u_ind).diff(t)

    #Dict to substitute variables when solving system  
    qr_zero = {qi: 0 for qi in system.q_dep}
    qd_zero = {qdi: 0 for qdi in q_d}
    ur_zero = {ui: 0 for ui in system.u_dep}
    # us_zero = {ui: 0 for ui in system.u_ind}
    # urd_zero = {udi: 0 for udi in u_dep_d}
    # usd_zero = {udi: 0 for udi in u_indep_d}

    #Solve kinematics differential equations for q_d
    Mk = kdes.jacobian(q_d)
    gk = kdes.xreplace(qd_zero)
    qd_sol = -Mk.LUsolve(gk)
    qd_repl = dict(zip(q_d, qd_sol)) #This further equations can be expressed without q_d
    
    #Solve velocity_constraints for dependant velocities
    vel_cons = vel_cons.xreplace(qd_repl)
    Mn = vel_cons.jacobian(system.u_dep)
    gn = vel_cons.xreplace(ur_zero)
    ur_sol = Mn.LUsolve(-gn)
    ur_repl = dict(zip(system.u_dep, ur_sol))
    
    #Solve velocity_constraints for dependant coordinates
    Mn = h_cons.jacobian(system.q_dep)
    gn = h_cons.xreplace(qr_zero)
    qr_sol = Mn.LUsolve(-gn)
    qr_repl = dict(zip(system.q_dep, qr_sol))
    
    #Replace dependant speed in kdes
    kdes = kdes.xreplace(ur_repl)
    
    # Differenciate vel_cons and replace dependant speed
    # fnd = vel_cons.diff(t).xreplace(qd_repl)
    
    #Solve for dependant generalized accelerations
    # Mnd = fnd.jacobian(u_dep_d)
    # gnd = fnd.xreplace(urd_zero).xreplace(ur_repl)
    # urd_sol = Mnd.LUsolve(-gnd)
    # urd_repl = dict(zip(u_dep_d, urd_sol))
    
    # me.find_dynamicsymbols(urd_sol)
    
    #Replace dependant variables in dynamical equations
    
    M_u4_u6_u7 = M_u4_u6_u7.xreplace(qd_repl)
    # me.find_dynamicsymbols(M_u4_u6_u7)
    
    F_u4_u6_u7 = F_u4_u6_u7.xreplace(qd_repl)
    F_u4_u6_u7 = F_u4_u6_u7.xreplace(ur_repl)
    F_u4_u6_u7 = F_u4_u6_u7.xreplace(qr_repl)

    print('Substitution sucessfully applied')
    
    expr_list = [M_u4_u6_u7, F_u4_u6_u7,
                 nh_cons, h_cons, kdes]
    
    names_list = ['M_u4_u6_u7', 'F_u4_u6_u7',
                 'nh_cons', 'h_cons', 'kdes']

    variable_list = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "u1",
        "u2",
        "u3",
        "u4",
        "u5",
        "u6",
        "u7",
        "u8",
        "steer_torque",
        "disturbance",
    ]

    generate_model_file("model_indep_dynamics", names_list, expr_list, variable_list, constants)

    # return M_u4_u6_u7, F_u4_u6_u7


def evaluation_casadi_file(constants: dict[str, float], x: np.ndarray, tau: float, distu: float):

    from model_files.model_d import (
        list_variables,
        list_constants,
        M_m,
        F_m,
    )

    M_m_inv = cas.inv(M_m)
    RHS = M_m_inv @ F_m

    f_RHS = cas.Function("RHS", list_variables + list_constants, [RHS])

    k = list(constants.values())

    RHS_num = np.array(f_RHS(*x.tolist() + [tau] + [distu] + k)).astype(float)

    print("CasADi RHS : ", RHS_num)


if __name__ == "__main__":

    x = np.ones((16,))
    tau = 1
    distu = 0

    system, constants, t = create_symbrim_model(simulation_flag=False, visualization_flag=False)

    export_constants(constants)

    # eval_num_full(system, constants, x, tau, distu)

    generate_casadi_file_indep_dynamics(system, constants)

    # evaluation_casadi_file(constants, x, tau, distu)
