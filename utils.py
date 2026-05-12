# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:04:33 2026

@author: Jules
"""

#From https://github.com/mechmotum/muscle-driven-bicycle-paper/blob/main/src/utils.py

import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

from IPython.display import HTML

from matplotlib.animation import FuncAnimation, HTMLWriter, PillowWriter, FFMpegWriter

from symbrim.utilities.plotting import Plotter
import pickle
import pandas as pd
import os
import seaborn as sns

from datetime import datetime

def make_new_results_folder(loc, suffixe=""):
    """
    make_new_results_folder


    """
    os.makedirs('results', exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    file_name = loc + f"{timestamp}/"
    if suffixe:
        file_name += f"_{suffixe}"
    
    try:
        os.makedirs(file_name, exist_ok=True)
        path = os.path.abspath(file_name)
        print(f"File generated : {path}")
        return file_name
    except OSError as e:
        print(f"Error when generating file : {e}")
        return None


def plot_trajectories(t, x, r, state_syms, input_syms, skip_first=False):

    q = x[0:14].copy()
    u = x[14:28].copy()
    if skip_first:
        q[:, 0] = np.nan
        u[:, 0] = np.nan
        r[:, 0] = np.nan
    q_sym = state_syms[0:14]
    u_sym = state_syms[14:28]

    qfig, qaxes = plt.subplots(7, 2, sharex=True)
    for i, (qi, ax, q_symi) in enumerate(zip(q, qaxes.T.flatten(), q_sym)):
        if i > 1:
            ax.plot(t, np.rad2deg(qi))
        else:
            ax.plot(t, qi)
        ax.set_ylabel(sm.latex(q_symi, mode='inline'))
    qaxes[-1, 0].set_xlabel('Time [s]')
    qaxes[-1, 1].set_xlabel('Time [s]')
    qfig.tight_layout()

    ufig, uaxes = plt.subplots(7, 2, sharex=True)
    for i, (ui, ax, u_symi) in enumerate(zip(u, uaxes.T.flatten(), u_sym)):
        if i > 1:
            ax.plot(t, np.rad2deg(ui))
        else:
            ax.plot(t, ui)
        ax.set_ylabel(sm.latex(u_symi, mode='inline'))
    uaxes[-1, 0].set_xlabel('Time [s]')
    uaxes[-1, 1].set_xlabel('Time [s]')
    ufig.tight_layout()

    rfig, raxes = plt.subplots(len(input_syms), 1, sharex=True)
    for ri, ax, r_symi in zip(r, raxes, input_syms):
        ax.plot(t, ri)
        ax.set_ylabel(sm.latex(r_symi, mode='inline'))
    ax.set_xlabel('Time [s]')
    rfig.tight_layout()
    
    

def plot_optimal_solution(t_simu, x_list, r_dep_list, r_ind_list, disturbance_list, n_start, model_index, sol_opt, path, save=True):

    fig, axs = plt.subplots(4,3)
    
    # print(len(sol_opt))
    # print(list(x_list[model_index]) + list(r_dep_list[0]) + list(r_ind_list[0]) + list(disturbance_list[model_index]))
    
    plot_config = [
          (0, 0, r'x_R [m]', 'conv_False'), # 'q1'
          (0, 0, r'y_R [m]', 'conv_False'), # 'q2'
          (0, 1, r'$\psi$ [deg]', 'conv_True'), # 'q3'
          (0, 1, r'$\varphi$ [deg]', 'conv_True'), # 'q4'
          (0, 1, r'$\theta$ [deg]', 'conv_True'), # 'q5'
          (0, 2, r'$\theta_R$ [deg]', 'conv_True'), # 'q6'
          (0, 1, r'$\delta$ [deg]', 'conv_True'), # 'q7'
          (0, 2, r'$\theta_F$ [deg]', 'conv_True'), # 'q8'
          (1, 0, r'$u$ [m/s]', 'conv_False'), # 'u1'
          (1, 0, r'$v$ [m/s]', 'conv_False'), # 'u2'
          (1, 1, r'$\dot{\psi}$ [deg/s]', 'conv_True'), # 'u3'
          (1, 1, r'$\dot{\varphi}$ [deg/s]', 'conv_True'), # 'u4'
          (1, 1, r'$\dot{\theta}$ [deg/s]', 'conv_True'), # 'u5'
          (1, 2, r'$\dot{\theta_R}$ [deg]', 'conv_True'), # 'u6'
          (1, 1, r'$\dot{\delta}$ [deg/s]', 'conv_True'), # 'u7'
          (1, 2, r'$\dot{\theta_F}$ [deg]', 'conv_True'), # 'u8'
          (2, 0, r'$T_{\delta}$ [Nm]', 'conv_False'), # 'Torques'
          (2, 1, r'$T_{\phi}$ [Nm]', 'conv_False'), # 'Torques'
          # (3, 0, r'K_steer []', 'conv_False'),
          # (3, 1, r'K_roll []', 'conv_False'),
          (2, 2, r'$T_{ped}$ [Nm]', 'conv_False'),
          (2, 0, r'$T_{\delta_{ff}}$ [Nm]', 'conv_False'),
          (3, 2, r'Disturbance [N]', 'conv_False')] 
    
    
    plot_config = {str(var)[:-3] : plot_config[n] for n, var in enumerate(list(x_list[model_index]) + list(r_dep_list[0]) + list(r_ind_list[0]) + list(disturbance_list[model_index]))}
    
    # print(plot_config)
    # print(len(sol_opt))
    
    for n, var in enumerate(list(x_list[model_index]) + list(r_dep_list[0]) + list(r_ind_list[0]) + list(disturbance_list[model_index])):
        fig.suptitle(f'Model {model_index}')
        i, j, label, conv = plot_config[str(var)[:-3]]

        if conv == 'conv_True':
            axs[i, j].plot(t_simu, np.rad2deg(sol_opt[n]), label = label)
        else:
            axs[i, j].plot(t_simu, sol_opt[n], label = label)
    
        
        axs[i, j].legend(bbox_to_anchor = (0.95, 1))
        plt.tight_layout()
    
    if save==True:
        plt.savefig(path + f'optimal_solution_for_model_{model_index}_start_{n_start}.png')
    

def animate_solution(t_simu, x_opt, T_opt, disturbance, bicycle, x, r, p, ani_name):

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
    param, param_vals = zip(*p.items())
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
    
    html_writer = HTMLWriter()
    ani.save(ani_name if ani_name.endswith(".html") else ani_name + ".html", writer=html_writer)

def save_results(to_save, path):
    
    model_type = to_save['model_type']
    n_start = to_save['n_start']
    
    file_to_save = path + f'solution_{model_type}_start_{n_start}'

    with open(f'{file_to_save}.pkl', 'wb') as file:
        pickle.dump(to_save, file)
    
def export_results_as_csv(t_simu, k_opt, x_opt, r_dep_opt, r_ind_opt, k, x, r, n_start, model_type, known_trajectories, path):
    
    x_and_r_dep_opt = np.vstack((x_opt, r_dep_opt))
    
    dict_for_pandas = { f'{var}' : x_and_r_dep_opt[i, :]  for i, (var, values) in enumerate(zip(x, x_and_r_dep_opt))}

    for j, var in enumerate(r):
        dict_for_pandas[f'{var}'] = r_ind_opt[j]
        
    dict_for_pandas.update(known_trajectories)
    
    dict_for_pandas['t_simu'] = t_simu
    dict_for_pandas['model_type'] = model_type
    dict_for_pandas['n_start'] = n_start
    
    for j, par in enumerate(k):
        dict_for_pandas[f'{par}'] = k_opt[j]
    
    df = pd.DataFrame(dict_for_pandas)
    
    fixed_cols = ['t_simu', 'model_type', 'n_start']
    data_cols = [c for c in df.columns if c not in fixed_cols]
    
    df_long = df.melt(id_vars=fixed_cols, value_vars=data_cols, var_name='var_raw', value_name='value')
    df_long['var_raw'] = df_long['var_raw'].fillna('').astype(str)

    extracted = df_long['var_raw'].str.extract(r'^m(\d+)_(.*)$')
    mask_success = extracted[0].notna()
    
    df_clean = df_long[mask_success].copy()
    
    if df_clean.empty:
        raise ValueError("Aucune colonne n'a pu être extraite. Vérifiez les noms de vos colonnes.")
    
    df_clean['model_index'] = extracted.loc[mask_success, 0].astype(int)
    df_clean['var_clean'] = extracted.loc[mask_success, 1].str.replace('(t)', '', regex=False)
    
    df_final = df_clean.pivot(
        index=fixed_cols + ['model_index'],
        columns='var_clean',
        values='value'
    ).reset_index()
    
    df_final.columns.name = None
    
    def sort_cols(prefix, cols):
        valid = [c for c in cols if str(c).startswith(prefix) and str(c)[len(prefix):].isdigit()]
        return sorted(valid, key=lambda x: int(str(x)[len(prefix):]))
    
    all_cols = df_final.columns.tolist()
    q_cols = sort_cols('q', all_cols)
    u_cols = sort_cols('u', all_cols)
    other_cols = ['K_steer','K_roll', 'pedaling_torque', 'disturbance']
    # Filtrer other_cols pour ne garder que ce qui existe
    other_cols = [c for c in other_cols if c in all_cols]
    
    final_order = fixed_cols + ['model_index'] + q_cols + u_cols + other_cols
    
    # Vérification finale avant réindexation
    missing = [c for c in final_order if c not in df_final.columns]
    if missing:
        print(f"Avertissement : Colonnes manquantes dans le résultat : {missing}")
        
    df_final = df_final[[c for c in final_order if c in df_final.columns]]
    
    df_final.to_csv(path + f'results_{model_type}_start_{n_start}.csv')
    
    
def stats_plot_about_same_model(model_type, PATH_to_results):
    

    list_results_csv_files = []
    
    for file in os.listdir(PATH_to_results):
        if file.endswith(".csv"):
            if model_type in file:
                list_results_csv_files.append(pd.read_csv(PATH_to_results+file))

    df = pd.concat(list_results_csv_files)    
    
    
    fig, axs = plt.subplots(2, 9)
    # axs = axs.flatten()
    
    for k, col in enumerate(['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']):
    
        sns.boxplot(df, y = col , x='model_index' , hue = 'n_start', ax = axs[0, k])
        
    for k, col in enumerate(['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8']):
        
        sns.boxplot(df, y = col , x='model_index' , hue = 'n_start', ax = axs[1, k])

    sns.boxplot(df, y = 'steer_torque' , x='model_index' , hue = 'n_start', ax = axs[0, 8])
    sns.boxplot(df, y = 'pedaling_torque' , x='model_index' , hue = 'n_start', ax = axs[1, 8])

