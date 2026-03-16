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