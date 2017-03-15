import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels as sm
import pylppl as lp
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import portfolio_functions as pf
import pylppl as lp

import matplotlib as mp

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 22}

label_size = 14
mp.rcParams['xtick.labelsize'] = label_size
mp.rcParams['ytick.labelsize'] = label_size

####### ####### ##### ###### ##### ########## # # # #
####                                             ####
####### ####### ##### ###### ##### ########## # # # #

def simple_OLS(pars, x):
    
    """OLS estimator for Beta reads
       beta_hat = (x'x)^-1 x'y.
    """
    
    beta = pars[0]
    alpha = 1.2
    
    epsilon = np.random.standard_normal(len(x))
    y       = np.random.standard_normal(len(x))
    
    for i in range(len(x)):
        y[i] = alpha + beta * x[i] + epsilon[i]
    
    return y

####### ####### ##### ###### ##### ########## # # # #
def check_cost_function(t1, t2, y, x, mod=False):
    
    """
    Check the cost function using statsmodel OLS for benchmarking
    -> Same of what Ive got
    """
    
    if mod == False:
        # RUN THE SIMPLE LOGIT REGRESSION
        logit_mod = sm.regression.linear_model.OLS(y[t1:t2], x[t1:t2])
        logit_res = logit_mod.fit()
        _ssr = logit_res.ssr/len(data)
        return _ssr
    else:
        # RUN THE SIMPLE LOGIT REGRESSION
        logit_mod = sm.regression.linear_model.OLS(y[t1:t2], x[t1:t2])
        logit_res = logit_mod.fit()
        _ssr = logit_res.ssr/len(data)
        return _ssr

####### ####### ##### ###### ##### ########## # # # #
def plot_2_ssr(SSR, SSR2, double=False):

    f,ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(SSR,linewidth=3, marker='s', color='b', markersize=10, markevery=4)
    ax.set_title('$SSR/N$ and $SSR/N - \lambda()$ of as a function of different sample sizes', fontsize=14)
    ax.set_xlabel(r'Sample size (N)',fontsize=20)
    ax.set_ylabel(r'$(SSR/N) - \lambda(t_2 - t_1)$',fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.grid(linewidth=.7)
    ax.legend([r'$DGP \rightarrow y = \beta X + \epsilon$'],loc='best',fontsize=20)
    if double == True:
        a = ax.twinx()
        a.plot(SSR2, linewidth=3, marker='o', color='k', markersize=10, markevery=4)
        #a.legend([r'$y = \beta X + \epsilon^2$'],loc='best',fontsize=20)
        a.tick_params(axis='both', which='major', labelsize=14)
        a.tick_params(axis='both', which='minor', labelsize=14)
        a.set_ylabel(r'$SSR/N \, (black)$',fontsize=22)
    else:
        pass
    plt.tight_layout()

####### ####### ##### ###### ##### ########## # # # #
def beta_OLS(y, x):
    
    m = np.shape(y)
    
    y = y.reshape(m[0],1)
    x = x.reshape(m[0],1)
    
    # Estimating beta OLS
    beta_hat = np.dot(x.T,x)**-1. * np.dot(x.T,y)
    
    # return
    return beta_hat

####### ####### ##### ###### ##### ########## # # # #
def get_sse_for_given_data(data, x, plot=False, standardize=False):
    
    # First we get beta
    beta = np.dot(x.T,x)**-1. * np.dot(x.T,data)
    
    # Pre alocate y
    y = []
    
    # recriate data
    for i in range(len(x)):
        y.append(beta * x[i])

    if plot == True:
        plt.plot(x, data, marker='o', linestyle='', markersize=8, color='r', alpha=0.9)
        plt.grid()
        plt.title(r'$\beta = %.3f$'%beta)
        plt.plot(x,y,color='k', linewidth=3.5)
    else:
        pass
    
    squared_residuals = (data - y)**2
    ssr = np.sum(squared_residuals)
    
    if standardize == False:
        return ssr
    else:
        return ssr/len(data)

####### ####### ##### ###### ##### ########## # # # #
def didier_cost(data, x, slope, standardize=True):
    
    # First we get beta
    beta = np.dot(x.T,x)**-1. * np.dot(x.T,data)
    
    # Pre alocate y
    y = []
    
    # recriate data
    for i in range(len(x)):
        y.append(beta * x[i])

    # get standardized sum of squares
    squared_residuals = (data - y)**2
    ssr_n = np.sum(squared_residuals)/len(data)
    ssr = np.sum(squared_residuals)
    lamb = - (ssr_n / len(data))
    ssr_new = ssr_n - lamb*(len(data))

    if standardize == True:
        return ssr_n - slope * (len(data))
    else:
        return ssr - (slope * (len(data)))

####### ####### ##### ###### ##### ########## # # # #
def get_sse_across_different_sample_sizes_OLS(y, x, slope, didier=False, plot=False, standardize=False):
    
    """ Check the Sum of Squared Residuals  
        at different sample-sizes
    """
    
    if didier == True:
    
        limit = len(y) - 10
    
        test = []
    
        for tt1 in range(limit):
            t1, t2 = tt1, -1
            test.append(didier_cost(y[t1:t2], x[t1:t2], slope, standardize))
    
        return test

    else:
        
        limit = len(y) - 10
        
        test = []
        
        for tt1 in range(limit):
            t1, t2 = tt1, -1
            test.append(get_sse_for_given_data(y[t1:t2], x[t1:t2], plot=plot, standardize=standardize))
        
        return test

####### ####### ##### ###### ##### ########## # # # #
def run_it_all_and_spit_normed_cost_according_to_didier(y, x, standardize=True):
    
    # calculate regularly the SSE/N for the linear model
    slope = 0.
    SSE = get_sse_across_different_sample_sizes_OLS(y, x, slope, standardize=standardize)
    
    # Fit the decreasing trend of the cost function
    slope = calculate_slope_of_normed_cost(SSE)
    
    # pass results into the iterator
    SSE_DS = get_sse_across_different_sample_sizes_OLS(y, x, slope[0], didier=True, standardize=standardize)
    
    return SSE_DS

####### ####### ##### ###### ##### ########## # # # #
def calculate_slope_of_normed_cost(sse):
    
    #Create linear regression object
    regr = linear_model.LinearRegression(fit_intercept=False)
    
    # create x range for the sse_ds
    x_sse = np.arange(len(sse))
    x_sse = x_sse.reshape(len(sse),1)
    
    # Train the model using the training sets
    res = regr.fit(x_sse, sse)
    
    return res.coef_

####### ####### ##### ###### ##### ########## # # # #
def simulate_and_plot_OLS_1(N, plot=True):

    """
    Define sample size and plot the normalized
    SSE as a function of the sample size**-1
    for the two different standardizations schemes,
    namely the original SSE/N and SSE/N - lambda(t2-t1)
    """

    # Simple monotonic increasing data vector
    x = np.arange(0,N,1)
    y = simple_OLS([0.5], x)

    slope=[]
    SSE = get_sse_across_different_sample_sizes_OLS(y,x,slope,plot=False,standardize=True)

    ds_sse = run_it_all_and_spit_normed_cost_according_to_didier(y, x, standardize=True)

    if plot == True:
        plot_2_ssr(ds_sse, SSE, double=True)
    else:
        return ds_sse, SSE


def OLS_solution(N=200):

    x = np.arange(0, N, 1)
    x[100:200] = x[100:200] * 2.
    y = simple_OLS([0.5], x)
    x = x + np.random.normal(0., 1., len(y))

    slope = []

    SSE = get_sse_across_different_sample_sizes_OLS(y, x, slope, plot=False, standardize=True)
    ds_sse = run_it_all_and_spit_normed_cost_according_to_didier(y, x, standardize=True)

    data = pd.DataFrame(y)

    return data, ds_sse, SSE

####### ####### ##### ###### ##### ########## # # # #
##          MORE COMPLICATED SCENARIO
####### ####### ##### ###### ##### ########## # # # #

def OLS_2(pars, x):

    """OLS estimator for Beta reads
       beta_hat = (x'x)^-1 x'y.
    """

    beta = pars[0]

    epsilon = np.random.standard_normal(len(x))
    y       = np.random.standard_normal(len(x))

    for i in range(len(x)):
        y[i] = beta * x[i] + epsilon[i]**3

    return y

####### ####### ##### ###### ##### ########## # # # #
#     LPPLS COST MODIFICATIONS CALCULATIONS
####### ####### ##### ###### ##### ########## # # # #
def get_lambda_from_normed_SSE(DF, use_normed=True):

    normed_chi_2 = DF.SSE.values / DF.dt.values

    if use_normed == True:
        y = normed_chi_2
    else:
        y = DF.SSE.values
    x = DF.dt.values

    # Fit the SSE
    mod = sm.regression.linear_model.OLS(y, x)
    mod_res = mod.fit()

    # Lambda
    _lambda = mod_res.params[0]

    return _lambda

def return_normalized_sse_and_normalized_sse_lambda(DF, _lambda, use_normed=True, index_t1=True):

    """
    if use_normed == True -> We use the normalised cost for calculations
    if index_t1 == True -> we return a dataframe with t1 as index and SSE/N(DS) values
    """

    if use_normed == True:
        # Get normed cost
        chi_2_normed = DF.SSE / DF.dt
    else:
        chi_2_normed = DF.SSE

    # Get chi2_normed acording to D.S.
    chi_2_normed_lambda = [(chi_2_normed[i] - _lambda*(DF.dt[i])) for i in range(len(chi_2_normed))]

    # Make t1's for the index
    if index_t1 == True:
        t1s = [pd.Timestamp(DF.index[0]) - np.int(DF.dt[i]) * pd.Timedelta('1D') for i in range(len(DF.index))]
        if use_normed == True:
            # Make-it a Data Frame
            ds_chi2 = pd.DataFrame(chi_2_normed_lambda, index = t1s, columns=[r'$(SSE/N) - \lambda()$'])
        else:
            # Make-it a Data Frame
            ds_chi2 = pd.DataFrame(chi_2_normed_lambda, index = t1s, columns=[r'$(SSE) - \lambda()$'])
    else:
        ds_chi2 = pd.DataFrame(chi_2_normed_lambda[::-1])

    if use_normed == True:
        # Returning the regular chi2 for comparsion
        reg_chi2 = pd.DataFrame((DF.SSE / DF.dt).values, index = t1s, columns=['SSE/N'])
    else:
        reg_chi2 = pd.DataFrame(DF.SSE.values, index = t1s, columns=['SSE'])

    return ds_chi2, reg_chi2

####### ####### ##### ###### ##### ########## # # # #
def estimate_and_plot_modified_SSE(res, data):

    # Get _lambda
    _lambda = get_lambda_from_normed_SSE(res, use_normed=False)
    ds_sse, reg_sse = return_normalized_sse_and_normalized_sse_lambda(res, _lambda, use_normed=False, index_t1=True)

    # Get non-normalised results
    _lambda = get_lambda_from_normed_SSE(res, use_normed=True)
    ds_sse2, reg_sse2 = return_normalized_sse_and_normalized_sse_lambda(res, _lambda, use_normed=True, index_t1=True)

    # plot
    plot_SSE_new(ds_sse, reg_sse, ds_sse2, reg_sse2, data)

####### ####### ##### ###### ##### ########## # # # #

def plot_SSE_new(ds_sse, reg_sse, ds_sse2, reg_sse2, data):

    f,ax = plt.subplots(2,2,figsize=(16,11))
    axs = ax.ravel()

    ####
    data[ds_sse.index[-1]:ds_sse.index[0]].plot(ax=axs[0], color='k', linewidth=3)
    axs[0].grid(True)
    axs[0].axvline(ds_sse[ds_sse==ds_sse.min()].dropna().index[0],
                   color='k', linewidth=4, linestyle='--')
    axs[0].set_xlabel('')

    ####
    data[ds_sse.index[-1]:ds_sse.index[0]].plot(ax=axs[1], color='k', linewidth=3)
    axs[1].grid(True)
    axs[1].axvline(ds_sse2[ds_sse2==ds_sse2.min()].dropna().index[0],
                       color='k', linewidth=4, linestyle='--')
    axs[1].set_xlabel('')

    ####
    ds_sse.plot(ax=axs[2], color='r', linewidth=3, marker='s', markersize=10)
    axs[2].axvline(ds_sse[ds_sse==ds_sse.min()].dropna().index[0],
                   color='k', linewidth=4, linestyle='--')
    a = axs[2].twinx()
    reg_sse.plot(ax=a, color='k', linewidth=3, marker='o', markersize=10)
    axs[2].set_yticks([])
    a.set_yticks([])
    axs[2].legend(loc='upper right', fontsize=14)
    a.legend(loc='upper center', fontsize=14)
    axs[2].grid()

    ####
    ds_sse2.plot(ax=axs[3], color='r', linewidth=3, marker='s', markersize=10)
    axs[3].axvline(ds_sse2[ds_sse2==ds_sse2.min()].dropna().index[0],
                   color='k', linewidth=4, linestyle='--')
    a = axs[3].twinx()
    reg_sse2.plot(ax=a, color='k', linewidth=3, marker='o', markersize=10)
    a.set_yticks([])
    axs[3].set_yticks([])
    axs[3].legend(loc='upper right', fontsize=14)
    a.legend(loc='upper center', fontsize=14)
    axs[3].grid()

    plt.tight_layout()

###################################
def estimate_and_plot_modified_SSE_NEW(res, data):
    # Get _lambda
    #_lambda = get_lambda_from_normed_SSE(res, use_normed=False)
    #ds_sse, reg_sse = return_normalized_sse_and_normalized_sse_lambda(res, _lambda, use_normed=False, index_t1=True)

    # Get non-normalised results! NB! USE_NORMED = True in order to work properly
    _lambda = get_lambda_from_normed_SSE(res, use_normed=True)
    ds_sse2, reg_sse2 = return_normalized_sse_and_normalized_sse_lambda(res, _lambda, use_normed=True, index_t1=True)

    # Plot Function (TO BE ADDED !!!)
    out1 = pf.normalize_data_for_comparacy(ds_sse2, sklearn=True)
    out2 = pf.normalize_data_for_comparacy(reg_sse2, sklearn=True)

    return out1, out2


###################################
def FinalPlot(data, res, ds_sse, reg_sse, t2):

    dahead = 250
    me = 3
    ms = 8
    lw = 1.7

    t1init = pd.Timestamp(t2) - 1200 * pd.Timedelta('1D')
    t1fin = pd.Timestamp(t2) + dahead * pd.Timedelta('1D')
    fakeIndex = pd.date_range(start=t2, end=t1fin)

    # FIT LPPLS ON THE Best DT
    bestT1 = ds_sse[ds_sse == ds_sse.min()]
    x = pd.Timestamp(t2) - pd.Timestamp(bestT1.dropna().index[0])
    x = x.days
    mres = lp.fit_series(data, t2, x)

    # Fake Axis
    newAxis = ds_sse.index.union(fakeIndex)
    na = pd.DataFrame(index=newAxis)
    ds_sse = pd.concat([na, ds_sse], axis=1)
    reg_sse = pd.concat([na, reg_sse], axis=1)


    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='col', sharey=False,
                             gridspec_kw={'height_ratios': [3, 2]},
                             figsize=(6, 5))

    axes[0].set_title(r'$t_1 = $%s; $t_2 =$ %s'%(str(bestT1.dropna().index[0])[0:10], t2))
    data[t1init:t2].plot(ax=axes[0], color='k')
    data[t1init:t1fin].plot(ax=axes[0], linewidth=0.3, color='k')
    axes[0].axvline(t2, color='k', linewidth=3)
    axes[0].set_ylabel(r'$\ln(P_t)$', fontsize=18)
    axes[0].axvline(bestT1.dropna().index[0],
                    color='k', linewidth=3, linestyle='--')
    fit = lp.lppl_project(mres, data, days=dahead)
    fit['fit'].plot(ax=axes[0], linewidth=2.5, color='r', alpha=0.7)
    fit['projection'].plot(ax=axes[0], linewidth=2.5, color='r',
                           linestyle=':')
    #axes[0].set_ylim([6.5, 9])
    #stdd = data[t1init:t1fin].std()

    #axes[0].set_ylim([data[t1init:t1fin].min().values[0]-stdd,
    #                  data[t1init:t1fin].max().values[0]+stdd])
    axes[0].legend('')
    plt.tight_layout()

    # Subplots
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': .0})
    axes[1].axvline(ds_sse[ds_sse == ds_sse.min()].dropna().index[0],
                    color='k', linewidth=3, linestyle='--')
    axes[1].axvline(t2, color='k', linewidth=3)
    ds_sse.plot(ax=axes[1], color='k', marker='^', markevery=me,
                markersize=ms, markerfacecolor='w',
                linewidth=lw)
    axes[1].set_ylabel(r'$\chi^2$', fontsize=18)
    a = axes[1].twinx()
    a.set_yticks([])
    reg_sse.plot(ax=a, color='k', marker='o', markevery=me,
                 markersize=ms, markerfacecolor='w',
                 linewidth=lw)
    axes[1].legend('')
    a.legend('')
    plt.tight_layout()
    plt.savefig('/Users/demos/Desktop/ttt'+str(t2)+'.pdf')


###################################
def FinalPlotOLS(data, ds_sse, reg_sse):

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='col', sharey=False,
                             gridspec_kw={'height_ratios': [2, 1]},
                             figsize=(7, 7))
    axes[0].plot(data)

    axes[1].plot(ds_sse)
    axes[1].plot(reg_sse)
    plt.tight_layout()


###################################
#     Post discussion with SP.
###################################
def generateSyntheticLppls(params=None, noise=0.01):

    """"
    generate synthetic LPPLS with given noise level with N = 1000
    """

    # params [tc, m, w, A, B, C1, C2]
    t = np.arange(0, 2000, 1)
    if params == None:
        params = [404.3, 0.748203, 6.226502,
                  1.762818, -7.263146e-04, 9.401988e-05,
                  8.613132e-05]
    else:
        params = params

    # Generate LPPLS
    sdata = lp.lppl(t, params, reduced=True)

    # Noise?
    noise = np.random.normal(0, noise, len(sdata))
    sdata = pd.DataFrame(sdata + noise)
    sdata.index = pd.date_range(start='1910-01-01', periods=len(sdata))

    return sdata















