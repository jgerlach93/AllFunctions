__author__ = 'demos'

# For parallel loops
from multiprocessing import cpu_count
try:
    CPU_COUNT = cpu_count()
except NotImplementedError:
    CPU_COUNT = 1

try:
    from joblib import Parallel, delayed
    RUN_PARALLEL = CPU_COUNT > 1
except ImportError:
    Parallel = None
    delayed = None
    RUN_PARALLEL = False

import sys
sys.path.append('/Users/demos/Documents/Python/ipy (work)/LPPLS - Sloppy/')
import sloppy_func as fsl
import numpy as np

import pylppl as lp
import pylppl_lik as lpl
import pandas as pd
import matplotlib.pyplot as plt

####################################################
def createContourPlotUsingModLikForGivenT2(Y, t2):

    dtRange  = np.arange(30, 750, 20)
    tcRange  = np.arange(-50, 200, 15)

    all_res = lp.construct_profile_matrix(Y, t2, dtRange, tcRange)

    good = lp.filter_profile_matrix(all_res, damp_min=0)

    return all_res, good


####################################################
def plotContour(all_res, good, ax=None):
    LI_level1=0.05
    LI_level2=0.5
    LI_level3=0.95

    prof = all_res['Lm']
    XX, YY = np.meshgrid(prof.columns, prof.index)
    XX2, YY2 = np.meshgrid(good.columns, good.index)

    if ax == None:
        alpha = 1
        plt.contourf(XX, YY, prof, [LI_level1, 1], colors='r', alpha=0.3)
        plt.contour(XX, YY, prof, [LI_level1, 1], colors='k', alpha=0.3)
        plt.contourf(XX, YY, prof, [LI_level2, 1], colors='r', alpha=0.6)
        plt.contour(XX, YY, prof, [LI_level2, 1], colors='k', alpha=0.6)
        plt.contourf(XX, YY, prof, [LI_level3, 1], colors='r')
        plt.contour(XX, YY, prof, [LI_level3, 1], colors='k')
        plt.contourf(XX2, YY2, good, [0, 0.1], colors='k', alpha=0.5)
        plt.axvline(0, color='k', linewidth=4)
    
        plt.ylabel(r'$t_1$', fontsize=22)
        plt.xlabel(r'$t_c$ $in$ $days$ ($t_2$ = $0$)', fontsize=22)
    else:
        ax.contourf(XX, YY, prof, [LI_level1, 1], colors='r', alpha=0.3)
        ax.contour(XX, YY, prof, [LI_level1, 1], colors='k', alpha=0.3)
        ax.contourf(XX, YY, prof, [LI_level2, 1], colors='r', alpha=0.6)
        ax.contour(XX, YY, prof, [LI_level2, 1], colors='k', alpha=0.6)
        ax.contourf(XX, YY, prof, [LI_level3, 1], colors='r')
        ax.contour(XX, YY, prof, [LI_level3, 1], colors='k')
        ax.contourf(XX2, YY2, good, [0, 0.1], colors='k', alpha=0.5)
        ax.axvline(0, color='k', linewidth=4)
        
        ax.set_ylabel(r'$t_1$', fontsize=22)
        ax.set_xlabel(r'$t_c$ $in$ $days$ ($t_2$ = $0$)', fontsize=22)
    
    plt.tight_layout()


####################################################
# Machine learning and the multi-scale LPPLS conf. indicator