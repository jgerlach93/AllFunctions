__author__ = 'demos'

import pylppl as lp
import scipy as sp
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/clean_codes/')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/LPPLS - Sloppy/')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/LPPLSDerivations/')

from sklearn.cluster import KMeans
import portfolio_functions as pf
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import data_functions as dfn
import sloppy_func as fsl

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

from numba import jit
import statsmodels as sm
import lpplsIndicators as lpi

#####################
### PLOTS DETAILS ###
#####################

import matplotlib as mp

font = {'family' : 'Helvetica',
    'weight' : 'normal',
        'size'   : 16}

label_size = 19
mp.rcParams['xtick.labelsize'] = label_size
mp.rcParams['ytick.labelsize'] = label_size

################################################################
@jit
def estimate_lppls(data,dt,t2):
    # fit model
    t, logP, _ = lp.prepare_data(data, t2, dt=dt)
    pars       = lp.fit_series(data, t2, dt=dt)

    return t, logP, pars
################################################################
def augmented_fit(data, t2, dt, fit_ABC=False):
    # estimate model!
    t, logP, pars = estimate_lppls(data,dt,t2)

    # estimate the sloppy degree
    sd, ph, rp, re = lp.test_sloppiness(t, logP, pars, fit_ABC=fit_ABC, pcombined=None)
    pars['sloppyness'] = sd
    pars['minimum'] = ph

    if fit_ABC is not True:
        for x in ['tc', 'm', 'w', 'A', 'B', 'C1', 'C2']:
            try:
                pars['i'+x] = rp.index(x)
            except:
                pars['i'+x] = np.NaN

        for x in ['tc', 'm', 'w', 'A', 'B', 'C1', 'C2']:
            pars['e'+x] = re[x]
    else:
        for x in ['tc', 'm', 'w']:
            try:
                pars['i'+x] = rp.index(x)
            except:
                pars['i'+x] = np.NaN

        for x in ['tc', 'm', 'w']:
            pars['e'+x] = re[x]

    return pars

################################################################
# PARALELIZING THE SEARCH OF T1
def run_search_t1_paralel(data, t2, fit_ABC=True, job_pool=None, lastDt=800):

    # Set dt range
    dt_range = np.linspace(30, lastDt, 60)

    # Run in parallel
    if RUN_PARALLEL:
        if job_pool is None:
            job_pool = Parallel(n_jobs=CPU_COUNT)
            res = job_pool(delayed(augmented_fit)(data, t2, dt, fit_ABC=fit_ABC) for dt in dt_range)

    # Return dataframe containing parameters
    _res = pd.DataFrame()
    for i in range(len(res)):
        _res = pd.concat([_res, res[i]], axis = 0)

    return _res

def run_search_t1_paralelV2(data, t2, fit_ABC=True, lastDt=800):
    """To be used only on house mac.
        For some unknown reason the RUN_PARALEL does not work on the little computer
        The Idea of the function is the same: estimate over serveral t1'
    """
    _res = pd.DataFrame()
    # Set dt range
    dt_range = np.linspace(30, lastDt, 60)
    for dt in dt_range:
        res = augmented_fit(data, t2, dt, fit_ABC=fit_ABC)
        _res = pd.concat([_res, res],axis=0)
    return _res

################################################################
def run_aug_dataframe_sev_t1s(data, t2, fit_ABC=False, dailybasis=False):

    # Estimate the model + parameters for a given DT and t2 !

    # pre-alocate dataFrame
    AUG_PARS = pd.DataFrame()

    if dailybasis == False:
        # t1 range
        dt = np.linspace(60, 750, 10).astype(np.int)
    else:
        dt = np.linspace(30, 750, 30)

    # start loop
    for DT in dt:
        aug_pars = augmented_fit(data, t2, DT, fit_ABC=fit_ABC)
        AUG_PARS = pd.concat([AUG_PARS,aug_pars], axis=0)

    return AUG_PARS
################################################################
def run_lppls_over_sev_t2s(data, t2, startt2, endt2, freq='M'):

    """estimate LPPLS over several t2's"""

    # set starting dates
    start = startt2
    end   = endt2
    if freq == 'M':
        T2S = pd.date_range(start=start, end=end, freq='M')
    else:
        T2S = pd.date_range(start=start, end=end, freq=pd.offsets.BDay(1))

    T2_RES = pd.DataFrame()
    for t2 in T2S:
        RES = run_aug_dataframe_sev_t1s(data, t2, fit_ABC=False)
        T2_RES = pd.concat([T2_RES,RES])

    return T2_RES
################################################################
def get_all_files_from_a_folder(path):
    # get files from the folder
    """ GET ALL H5 files from a folder """
    
    FILES = dfn.getFiles(path)

    # append results
    HUGE = pd.DataFrame()
    K = FILES.keys()[:]; K = np.sort(K)

    # Run loop
    for files in K:
        try:
            DF = pd.read_hdf(path+files,'res')
            HUGE = pd.concat([HUGE,DF])
        except:
            pass

    # Fill na with zeros
    HUGE = HUGE.fillna(0)
    
    return HUGE
################################################################
def filter_big_df(HUGE):
    """ filter everyThingBro """
    cond = HUGE.ix[(HUGE['m']>0.1) & (HUGE['m']< 0.9)] 
    cond1 = cond.ix[(cond['w']>6.0) & (cond['w']< 13)]
    cond2 = cond1.ix[cond1['minimum'] == 1]

    s,z   = np.shape(HUGE)
    ss,zz = np.shape(cond2)
    print('HUGE before filter size = %s'%s)
    print('HUGE after filter size = %s'%ss)
    
    return cond2
################################################################
def count_it(x, num=3):
    inds = range(num)
    return pd.Series([(x==i).sum() for i in inds], index=inds)

def plot_it(df):
    if 'iA' in df:
        f, axs = plt.subplots(1,7, figsize=(12,4))
        axs = axs.ravel()
        keys = ['itc','im','iw','iA','iB','iC1','iC2']
    else:
        f, axs = plt.subplots(1,3, figsize=(8,3))
        keys = ['itc','im','iw']

    for key,ax in zip(keys, axs):
        x = count_it(df[key], num=len(keys))/len(df)
        x.plot(kind='barh', color='k', ax=ax)

        ax.set_xlim([0,1])
        ax.set_ylim([-0.5,len(keys)-0.5])
        ax.set_ylabel('')
        ax.set_title(key[1:])
    if 'iA' in df:
        plt.sca(axs[-1])
        plt.axis('off')

    plt.tight_layout()

def plot_it_e(df):
    if 'iA' in df:
        keys1 = ['etc','em','ew','eA','eB','eC1','eC2']
        keys2 = ['itc','im','iw','iA','iB','iC1','iC2']
        rng = 15.
        fx = 15
    else:
        keys1 = ['etc','em','ew']
        keys2 = ['itc','im','iw']
        rng = 6.
        fx = 7
    f, axs = plt.subplots(2,len(keys1), figsize=(fx,5))

    for k1,k2,ax in zip(keys1, keys2, np.transpose(axs)):
        x = -np.log10(df[k1])
        x.hist(bins=np.linspace(0, rng, 100), color='b', ax=ax[1], normed=True)
        ax[1].set_xlim([0, rng])
        ax[1].set_xlim([-0.5, rng+0.5])
        ax[1].set_ylim([0, 1])

        x = pd.Series(count_it(df[k2], num=len(keys1))/len(df))
        x.plot(kind='bar', color='b', ax=ax[0])
        ax[0].set_ylim([0,1])
        ax[0].set_xlim([-0.5,len(keys1)-0.5])
        ax[0].set_xticks(range(len(keys1)))
        ax[0].set_title(k1[1:])

        if not(k1=='etc'):
            ax[0].set_yticklabels([])
            ax[1].set_yticklabels([])

    plt.tight_layout()
    #plt.savefig('sloppy_hist_3.pdf')

################################################################
def plot_fig_sloppy_vs_index(test,data,DF):
    
    """
    Create plot fig2. sloppiness vs time series + histogram on the upper left corner
    """

    y1,y2 = '1991','2016'
    f,ax = plt.subplots(1,1,figsize=(18,5))
    np.abs(test[y1:y2].sloppyness).plot(kind='area',ax=ax,color='b',linewidth=0.5,alpha=0.2)
    np.abs(test[y1:y2].sloppyness).plot(ax=ax,color='b',marker='o',markersize=5,linewidth=0.5,alpha=0.2)
    ax.set_ylabel('$\Lambda$'); 
    #ax.set_ylim([0,15])
    ax.set_ylim([0,22])
    a = ax.twinx()
    data[y1:y2].plot(grid=False,ax=a,color='k',linewidth=2)
    a.set_ylabel('$\ln[P(t)]$')
    plt.tight_layout()

    #####
    aa = plt.axes([0.11, 0.7, .2, .2])
    sns.distplot(DF.sloppyness,color='b',label='',ax=aa)
    aa.set_ylabel('$pdf(\Lambda)$')
    aa.set_xlabel('$\Lambda$')
    aa.axvline(np.mean(DF.sloppyness),color='r')
    aa.axvspan(np.mean(DF.sloppyness)-np.std(DF.sloppyness),
               np.mean(DF.sloppyness)+np.std(DF.sloppyness),
               color='r',alpha=0.2)
    plt.tight_layout()

################################################################
def plot_fig_sloppy_vs_index_2(test,data,DF,test2,DF2):
    y1,y2 = '1991','2016'
    f,ax = plt.subplots(1,1,figsize=(18,5))
    np.abs(test[y1:y2].sloppyness).plot(kind='area',ax=ax,color='b',linewidth=0.5,alpha=0.2)
    np.abs(test[y1:y2].sloppyness).plot(ax=ax,color='b',marker='o',markersize=3,linewidth=0.5,alpha=0.2)
    np.abs(test2[y1:y2].sloppyness).plot(kind='area',ax=ax,color='r',linewidth=0.5,alpha=0.2)
    np.abs(test2[y1:y2].sloppyness).plot(ax=ax,color='r',marker='s',markersize=3,linewidth=0.5,alpha=0.2)
    ax.set_ylabel('$\Lambda$');
    ax.set_ylim([0,25])
    a = ax.twinx()
    data[y1:y2].plot(grid=False,ax=a,color='k',linewidth=2)
    a.set_ylabel('$\ln[P(t)]$')
    plt.tight_layout()

    #####
    aa = plt.axes([0.11, 0.7, .2, .2])
    sns.distplot(DF.sloppyness,color='b',label='',ax=aa)
    sns.distplot(DF2.sloppyness,color='r',label='',ax=aa)
    aa.set_ylabel('$pdf(\Lambda)$')
    aa.set_xlabel('$\Lambda$')
    aa.axvline(np.mean(DF.sloppyness),color='b')
    aa.axvspan(np.mean(DF.sloppyness)-np.std(DF.sloppyness),
               np.mean(DF.sloppyness)+np.std(DF.sloppyness),
               color='b',alpha=0.2)
    aa.axvline(np.mean(DF2.sloppyness),color='r')
    aa.axvspan(np.mean(DF2.sloppyness)-np.std(DF2.sloppyness),
               np.mean(DF2.sloppyness)+np.std(DF2.sloppyness),
               color='r',alpha=0.2)
    #aa.set_xlim([0,25])
    plt.tight_layout()

################################################################
def generate_synthetic_lppls_and_noise(sigma):

    tc,m,w,A,B,C,D = [1194,0.44,6.500112,1.826598,-0.00944,-0.000189,0.0005];
    pars = [tc,m,w,A,B,C,D]
    t = np.arange(0,1500,1)
    y=lp.lppl(t,pars,reduced=True);

    # generate noise
    yn = lp.np.random.normal(0,sigma,len(t))

    # data range
    start =  lp.pd.datetime(1990, 1, 1)
    rng = lp.pd.date_range(start,periods=len(t),freq='d'); rng=rng.values.astype('M8[D]')
    df = lp.pd.DataFrame(y+yn,columns=['P(t)'])
    df.index = rng

    # plot
    df.plot(color='k',marker='.',linestyle='',figsize=(8,3)); plt.ylim([min(y+yn),max(y+yn)])
    plt.show()

    return df

################################################################
class AR(object):
    def __init__(self, data=None, ar_order=0):
        self.data = data
        self.ar_order = ar_order
        self.parameters = None
        self.sigma2 = 1

    def simulate(self, T=1400):
        if self.parameters is None:
            raise ValueError('Parameters must be estimated or specified before simulation')
        ar_order = self.ar_order
        tau = T + self.ar_order
        e = np.random.standard_normal((tau,))
        y = np.zeros((tau,))
        sigma = np.sqrt(self.sigma2)
        const = self.parameters[0]
        ar_params = self.parameters[1:]
        if ar_order == 0:
            return const + sigma * e

        # A solution to handle complexities of indexing near 0
        y[ar_order] = const + ar_params.dot(y[ar_order-1::-1]) + e[ar_order]
        for t in xrange(ar_order + 1,tau):
            y[t] = const + ar_params.dot(y[t-1:t-ar_order-1:-1]) + e[t]

        return y[ar_order:]

    def estimate(self):
        if self.data is None:
            raise ValueError('Data must be provided to estimate parameters')

        T = self.data.shape[0]
        y = self.data[:,None]
        ar_order = self.ar_order
        tau = T - self.ar_order
        x = np.ones((tau, ar_order + 1))
        for i in xrange(ar_order):
            x[:, [i+1]] = y[ar_order-1-i:T-i-1]

        xpxi = np.linalg.pinv(x)
        self.parameters = xpxi.dot(y[ar_order:T])
        self.errors = y[ar_order:T]-x.dot(self.parameters)
        e = self.errors
        self.sigma2 = e.T.dot(e) / tau
        self.sigma2 = self.sigma2.squeeze()

    def forecast(self, in_sample=None):
        # TODO:
        raise NotImplementedError("Forecast has not been implemented!")

################################################################

################################################################
def status_indicator_at_t2(t2, DF, pos=True, fromQz=False):

    """ return the status indicator at a given t2
        - give dataframe of estimated parameters DF
        - give t2.
        - choose between pos and neg bubble
        What we do here is: return a scalar value := DS TRUST INDICATOR
        after filtering fits according to some rules.
        """

    # get dataframe Shape
    x = np.shape(DF.ix[t2])

    # Current dataframe
    cdf = DF.ix[t2]

    # Check if it is or not a dataframe first
    if isinstance(cdf,pd.DataFrame):
        if fromQz is not True:

            # filter IF POSITIVE BUBBLES
            if pos == True:
                cdf = cdf.ix[cdf['m']<0.9]
                cdf = cdf.ix[cdf['m']>0.1]
                cdf = cdf.ix[cdf['B']<0.]
                cdf = cdf.ix[cdf['tc']>=0]
            # IF WE WANT NEGATIVE ONLY
            else:
                cdf = cdf.ix[(cdf['m']<0.9)]
                cdf = cdf.ix[cdf['m']>0.1]
                cdf = cdf.ix[cdf['B']>=0.]
                cdf = cdf.ix[cdf['tc']>=0]

            # size of dataframe post-filtering
            x_p = np.shape(cdf)

            # indicator
            res = x_p[0]/np.float(x[0])

            return res

        else:

            # filter IF POSITIVE BUBBLES
            if pos == True:
                cdf = cdf.ix[cdf['_m']<0.9]
                cdf = cdf.ix[cdf['_m']>0.1]
                cdf = cdf.ix[cdf['_B']<0.]
                cdf = cdf.ix[cdf['_tc']>=t2]
            # IF WE WANT NEGATIVE ONLY
            else:
                cdf = cdf.ix[(cdf['_m']<0.9)]
                cdf = cdf.ix[cdf['_m']>0.1]
                cdf = cdf.ix[cdf['_B']>=0.]
                cdf = cdf.ix[cdf['_tc']>=t2]

            # size of dataframe post-filtering
            x_p = np.shape(cdf)

            # indicator
            res = x_p[0]/np.float(x[0])

            return res

    else:
        return 0.0

################################################################
def create_lppls_confidence_indicator(DF,startt2,endt2,t2,hierarchy=False):
    """Run over several t2s and construct the indicator (pos and neg)
        - Here we run across several t2s and estimate the LPPLS model.
        - We also calculate either the hierarchical indicator or the regular one
        - Func built on July.11.16
    """

    # Run over several t2's
    result = fsl.run_lppls_over_sev_t2s(DF, t2, startt2, endt2, freq='M')

    # Filter
    result_f = filter_big_df(result)

    # construct the indicator
    # Transform dt into timestamp
    result_f['dt_1'] = result_f.index + (-result_f.dt) * pd.Timedelta('1D')

    # Transform tc into timestamp
    result_f['tc_1'] = result_f.index + (result_f.tc.astype(int)) * pd.Timedelta('1D')

    # Get all t2s
    t2s = result_f.index.unique()

    ## construct
    # Iterate over all t2's
    # Construct positive and negative bbble indicators
    if hierarchy == True:
        X_pos = [status_indicator_at_t2_hiearchy(t2,result_f) for t2 in t2s]
        X_neg = [status_indicator_at_t2_hiearchy(t2,result_f,pos=False) for t2 in t2s]
        _trust = pd.DataFrame(X_pos,index=t2s,columns=['lrg','med','shrt'])
        _trust_neg = pd.DataFrame(X_neg,index=t2s,columns=['lrg','med','shrt'])

        # Merge with data
        pos_merged = pd.concat([DF,_trust],axis=1);
        pos_merged = pos_merged.fillna(0)
        neg_merged = pd.concat([DF,_trust_neg],axis=1);
        neg_merged = neg_merged.fillna(0)

    else:
        
        X_pos = [status_indicator_at_t2(t2,result_f) for t2 in t2s]
        X_neg = [status_indicator_at_t2(t2,result_f,pos=False) for t2 in t2s]
        _trust = pd.DataFrame(X_pos,index=t2s,columns=['indicator'])
        _trust_neg = pd.DataFrame(X_neg,index=t2s,columns=['indicator'])

    return _trust, _trust_neg

################################################################
def plot_indicator_vs_ts(data, _trust, _trust_neg, thr_pos, thr_neg, y1, y2):
    # visual inspection of the negative bubble signals
    f,ax = plt.subplots(1,1,figsize=(11,4))
    data.ix[_trust.index][y1:y2].plot(color='k',ax=ax)
    a = ax.twinx()
    (_trust[y1:y2]>=thr_pos).plot(kind="area",ax=a,color='r',alpha=0.5)
    (_trust_neg[y1:y2]>=thr_neg).plot(kind="area",ax=a,color='g',alpha=0.5)
    plt.tight_layout()
################################################################
def plot_indicator_vs_ts_sevTrh(data, _trust, _trust_neg, y1, y2):

    # set the thresholds
    _thr = [0.1,0.2,0.3,0.4,0.5,0.6]

    # visual inspection of the negative bubble signals
    f,ax = plt.subplots(6,1,figsize=(11,24))
    axs = ax.ravel()
    for i in range(len(_thr)):
        data.ix[_trust.index][y1:y2].plot(color='k',ax=axs[i])
        a = axs[i].twinx()
        (_trust[y1:y2]>=_thr[i]).plot(kind="area",ax=a,color='r',alpha=0.5)
        (_trust_neg[y1:y2]>=_thr[i]).plot(kind="area",ax=a,color='g',alpha=0.5)
        axs[i].set_title('Threshold = %s'%_thr[i])
    plt.tight_layout()
################################################################
def mean_ret_after_a_signal(data,t2,debug=False):

    """
    :param data: time-series
    :param t2: observation that the conditional returns will be calculated
    :return: dataframe with the mean(rt(t2:t2+delta_i) for i in days
    """

    # Pre-alocate
    _res = []

    if debug == False:
        # Days after t2 for checking conditional returns
        days = np.arange(10,200,10)

        # get mean returns
        for delta in days:
            res = data[t2:(t2 + pd.offsets.BDay(delta))].pct_change().mean()[0]*100
            _res.append(res)

        return pd.DataFrame(_res, index=days, columns=[str(t2)[0:10]])
    else:
        # Days after t2 for checking conditional returns
        days = np.arange(10,200,10)

        # get mean returns
        for delta in days:
            res = (data[t2:(t2 + pd.offsets.BDay(delta))].pct_change().dropna()*100).quantile(.25)[0]
            _res.append(res)

        return pd.DataFrame(_res, index=days, columns=[str(t2)[0:10]])

################################################################
def get_cond_ret_dataframe(signal, data, debug=False):

    """
    :param signal: boolean dataframe (F,T) whose index is our focus
    :param data: ts
    :return: concatenate over |mean_ret_after_a_signal|

    -> Cuidar pois, to indo ate o len(data) para calcular os cond ret.
       isso vai dar pau visto que nao posso calcular 100 dias depois da minha ultima observacao.
    """

    X = pd.DataFrame()

    if debug == False:
        for i in range(len(signal)):
            if signal.ix[i][0] == False:
                pass
            else:
                t2_to_use =signal.index[i]
                O = mean_ret_after_a_signal(data,t2_to_use)
                X = pd.concat([X,O],axis=1)

        return X.T
    else:
        for i in range(len(data)):
            t2_to_use =data.index[i]
            O = mean_ret_after_a_signal(data, t2_to_use, debug=debug)
            X = pd.concat([X,O],axis=1)

        return X.T

################################################################
def figure_boxplot_neg_and_pos(pos_conditional_res_df,neg_conditional_res_df):
    f,ax = plt.subplots(1,2,figsize=(10,4))
    pos_conditional_res_df.boxplot(bootstrap=True, whis=np.inf, ax=ax[0])
    ax[0].set_title('positive bubbles')
    ax[1].set_title('negative bubbles')
    neg_conditional_res_df.boxplot(bootstrap=True, whis=np.inf, ax=ax[1])
    ax[0].set_xlabel('$\Delta$ in days after t_2'); ax[0].set_ylabel('$\mu(r(t_2:t_2+/delta_i))$')
    ax[1].set_xlabel('$\Delta$ in days after t_2');
    plt.tight_layout()

################################################################
def test_different_threshold(_trust,_trust_neg,data,y1,y2,modified_sig=False):
    # Loop results over several thresholds

    X_POS = []
    X_NEG = []

    if modified_sig == False:
        v = [0.1,0.2,0.3,0.4,0.5,0.6]
    else:
        v = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    for i in v:
        _cond_returns_pos = get_cond_ret_dataframe((_trust[y1:y2]>=i),data)
        X_POS.append(_cond_returns_pos)
        _cond_returns_neg = get_cond_ret_dataframe((_trust_neg[y1:y2]>=i),data)
        X_NEG.append(_cond_returns_neg)

    return X_POS, X_NEG, v
################################################################
def plot_auch_thresholds(X_POS, X_NEG, v, boxplot=False):
    # plot
    f,ax = plt.subplots(2,6,figsize=(36,11))
    axs = ax.ravel()

    if boxplot is not True:
        try:
            for i in range(6):
                sns.violinplot(X_POS[i], bw=0.02,ax=axs[i])
                #axs[i].set_ylim([-1.,1])
                axs[i].set_title('thr = %s'%v[i])

            for i in range(6):
                sns.violinplot(X_NEG[i], bw=0.02,ax=axs[i+6])
                plt.tight_layout()
        except:
            pass
    else:
        try:
            for i in range(6):
                X_POS[i].boxplot(whis=np.inf,ax=axs[i])
                axs[i].set_title('thr = %s'%v[i])
                axs[i].axhline(0,color='k',linestyle='-')

            for i in range(6):
                X_NEG[i].boxplot(whis=np.inf,ax=axs[i+6])
                axs[i+6].axhline(0,color='k',linestyle='-')
                #axs[i].set_ylim([-1.,1])
                plt.tight_layout()
        except:
            pass

    axs[6].set_ylabel('Negative bubble')
    axs[0].set_ylabel('Positive bubble')

    plt.tight_layout()

################################################################
def plot_cond_ret_after_a_sig(signal, data):

    """ USE DIFFERENT SIGNAL THAN THE INDICATOR
        and then boxplot
    """

    #calculate
    _COND = pd.DataFrame()

    for t2 in signal.index:
        _cond = mean_ret_after_a_signal(data,t2)
        _COND = pd.concat([_COND,_cond],axis=1)

    # conditional on a qualified t2, check returns at t2+10, ... t2+210.
    _COND.T.boxplot()
    plt.tight_layout()
################################################################
def check_cond_rets_over_different_t2s(signal,data,days):

    """
    :param signal: signal (what matters is the index)
    :param days: scalar number containing the number of days for checking conditional returns
    :return:
        - conditional returns from t2 to tc -> A *** MEAN
        - conditional returns from t2 to t2+days -> B
        - conditional returns from t2 to tc+30 -> We can add the uncertatinty estimation here
    """

    # prealocate
    A = []
    B = []
    C = []

    # test cool
    for i in range(len(signal)):
        first_ob = signal.index[i]
        last_ob  = signal.ix[i].tc_1 # Choose if we want the tc data
        last_ob2 = first_ob + pd.offsets.BDay(days) # Trinta dias after t2 is good!

        # Get mean return within this period (t2 until tc)
        A.append((data[first_ob:last_ob].diff().mean()*100)[0])#

        # Get mean return within this period (t2 until tc)
        B.append((data[first_ob:last_ob2].diff().mean()*100)[0])

        # Get mean return within this period (tc until tc+30)
        C.append((data[first_ob:last_ob + pd.offsets.BDay(30)].diff().mean()*100)[0])

    # Compile everything into a dataframe
    cond_ret_t2_tc            = pd.DataFrame(A,index=[signal.index]).dropna()
    cond_ret_t2_plus_30       = pd.DataFrame(B,index=[signal.index]).dropna()
    cond_ret_t2_tc_plus_30    = pd.DataFrame(C,index=[signal.index]).dropna()

    return cond_ret_t2_tc, cond_ret_t2_plus_30, cond_ret_t2_tc_plus_30

################################################################
def new_plot_boxplot_3_3(thrs_res_pos,v):

    f,ax = plt.subplots(3,3,figsize=(15,10),sharey=True, sharex=True)
    axs = ax.ravel()
    for i in range(9):
        bp = thrs_res_pos[i].boxplot(notch=0, sym='+', vert=1, whis=1.5, ax=axs[i])
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')
        axs[i].set_title('thr = %s'%v[i],fontsize=14)
        axs[i].axhline(0,color='k',linestyle='-')
        #axs[i].set_ylim([-8,8])
        axs[i].tick_params(axis='both', which='major', labelsize=16)
        axs[i].tick_params(axis='both', which='minor', labelsize=16)
    axs[0].set_ylabel(r'$r_t$',fontsize=30); axs[3].set_ylabel(r'$r_t$',fontsize=18);
    axs[6].set_ylabel(r'$r_t$',fontsize=18)
    [axs[i].set_xlabel(r'$Days$',fontsize=18) for i in [6,7,8]]
    [plt.setp( axs[i].xaxis.get_majorticklabels(), rotation=70, horizontalalignment='right' ) for i in [6,7,8]]
    plt.tight_layout()

################################################################
def status_indicator_at_t2_hiearchy(t2, DF, pos=True,fromQz=False):

    """ return the status indicator at a given t2
        - give dataframe of estimated parameters DF
        - give t2.
        - choose between pos and neg bubble
        What we do here is: return a scalar value := DS TRUST INDICATOR
        after filtering fits according to some rules.
        """

    # get dataframe Shape
    x = np.shape(DF.ix[t2])

    # Current dataframe
    cdf = DF.ix[t2]

    # Check if it is or not a dataframe first
    if isinstance(cdf,pd.DataFrame):
        if fromQz is not True:

            # filter IF POSITIVE BUBBLES
            if pos == True:
                cdf = cdf.ix[cdf['m']<0.9]
                cdf = cdf.ix[cdf['m']>0.1]
                cdf = cdf.ix[cdf['B']<0.]
                cdf = cdf.ix[cdf['tc']>=0]
            # IF WE WANT NEGATIVE ONLY
            else:
                cdf = cdf.ix[(cdf['m']<0.9)]
                cdf = cdf.ix[cdf['m']>0.1]
                cdf = cdf.ix[cdf['B']>=0.]
                cdf = cdf.ix[cdf['tc']>=0]

            # size of dataframe post-filtering
            x_p = np.shape(cdf)

            # indicator
            res = x_p[0]/np.float(x[0])

            return count_number_of_timeScales_qualified(cdf)

        else:

            # filter IF POSITIVE BUBBLES
            if pos == True:
                cdf = cdf.ix[cdf['_m']<0.9]
                cdf = cdf.ix[cdf['_m']>0.1]
                cdf = cdf.ix[cdf['_B']<0.]
                cdf = cdf.ix[cdf['_tc']>=t2]
            # IF WE WANT NEGATIVE ONLY
            else:
                cdf = cdf.ix[(cdf['_m']<0.9)]
                cdf = cdf.ix[cdf['_m']>0.1]
                cdf = cdf.ix[cdf['_B']>=0.]
                cdf = cdf.ix[cdf['_tc']>=t2]

            # size of dataframe post-filtering
            x_p = np.shape(cdf)

            # indicator
            res = x_p[0]/np.float(x[0])

            return count_number_of_timeScales_qualified(cdf)

    else:
        return np.array([0,0,0])

################################################################
def count_number_of_timeScales_qualified(cdf):

    # Get dts
    dts_s = cdf.dt

    # total points per time_scale of interest
    total_shrt  = 9.
    total_med   = 10.
    total_large = 14.

    # intervales
    shrt  = np.linspace(60,200,(200-60)).astype(int)
    med   = np.linspace(200,400,(400-200)).astype(int)
    large = np.linspace(400,700,(700-400)).astype(int)

    #
    lrg_append, shrt_append, med_append = [],[],[]
    for i in range(len(cdf)):
        lrg_append.append(cdf.dt[i] in large)
        med_append.append(cdf.dt[i] in med)
        shrt_append.append(cdf.dt[i] in shrt)

    # compile
    total_largeScale_sigs = np.sum(lrg_append)/total_large
    total_medScale_sigs = np.sum(med_append)/total_med
    total_shrtScale_sigs = np.sum(shrt_append)/total_shrt



    return np.array([total_largeScale_sigs, total_medScale_sigs, total_shrtScale_sigs])

################################################################
def count_number_of_timeScales_qualified_fine_grid(cdf):

    # Get dts
    dts_s = cdf.dt

    # intervales
    shrt  = np.linspace(30,90,(90-30)).astype(int)
    med   = np.linspace(90,300,(300-90)).astype(int)
    large = np.linspace(300,745,(745-300)).astype(int)

    # total points per time_scale of interest
    total_shrt  = np.str(len(shrt))/5.
    total_med   = np.str(len(med))/5.
    total_large = np.str(len(large))/5.

    #
    lrg_append, shrt_append, med_append = [],[],[]
    for i in range(len(cdf)):
        lrg_append.append(cdf.dt[i] in large)
        med_append.append(cdf.dt[i] in med)
        shrt_append.append(cdf.dt[i] in shrt)

    # compile
    total_largeScale_sigs = np.sum(lrg_append)/total_large
    total_medScale_sigs = np.sum(med_append)/total_med
    total_shrtScale_sigs = np.sum(shrt_append)/total_shrt



    return np.array([total_largeScale_sigs, total_medScale_sigs, total_shrtScale_sigs])

################################################################
def G_trust(ag,_trust,shrt=False,med=False,lrg=False,plot=True):

    if plot == True:
        if shrt is not False:
            # Trust_green
            TERM_SHRT = 1/(1-0.9801)
            tg = _trust['shrt'].copy()
            G = np.repeat(np.mean(tg),len(tg))
            for t in range(len(tg)):
                if t == 0:
                    G[t] = tg[0]
                else:
                    G[t] = ag * G[t-1] + tg[t-1]
            head = 'shrt'

        elif med is not False:
            # Trust_orange
            TERM_MED = 1/(1-0.99501)
            tg = _trust['med'].copy()
            G = np.repeat(np.mean(tg),len(tg))
            for t in range(len(tg)):
                if t == 0:
                    G[t] = tg[0]
                else:
                    G[t] = ag * G[t-1] + tg[t-1]
            head = 'med'

        elif lrg is not False:
            # Trust_red
            TERM_LONG = 1/(1-0.998001)
            tg = _trust['lrg'].copy()
            G = np.repeat(np.mean(tg),len(tg))
            for t in range(len(tg)):
                if t == 0:
                    G[t] = tg[0]
                else:
                    G[t] = ag * G[t-1] + tg[t-1]
            head = 'lrg'

        # index df
        DF = pd.DataFrame(G,index=tg.index,columns=[head])

        # ca
        hub = (1/(1.-ag))

        # plot
        f,ax = plt.subplots(1,1,figsize=(10,4))
        DF.plot(ax=ax)
        a = ax.twinx()
        pos_merged['P/D'].plot(color='k',ax=a)
        a.set_title('parameter = %s ; (1/1-ag) = %.2f days'%(ag,hub))
        plt.tight_layout()
    else:
        if shrt is not False:
            # Trust_green
            TERM_SHRT = 1/(1-0.9801)
            tg = _trust['shrt'].copy()
            G = np.repeat(np.mean(tg),len(tg))
            for t in range(len(tg)):
                if t == 0:
                    G[t] = tg[0]
                else:
                    G[t] = 0.9801 * G[t-1] + tg[t-1]
            head = 'shrt'

        elif med is not False:
            # Trust_orange
            TERM_MED = 1/(1-0.99501)
            tg = _trust['med'].copy()
            G = np.repeat(np.mean(tg),len(tg))
            for t in range(len(tg)):
                if t == 0:
                    G[t] = tg[0]
                else:
                    G[t] = 0.99501 * G[t-1] + tg[t-1]
            head = 'med'

        elif lrg is not False:
            # Trust_red
            TERM_LONG = 1/(1-0.998001)
            tg = _trust['lrg'].copy()
            G = np.repeat(np.mean(tg),len(tg))
            for t in range(len(tg)):
                if t == 0:
                    G[t] = tg[0]
                else:
                    G[t] = 0.998001 * G[t-1] + tg[t-1]
            head = 'lrg'

        # index df
        DF = pd.DataFrame(G,index=tg.index,columns=[head])

        return DF

################################################################
# plotting
################################################################
def plot_box_at_thr(thr,pos_conditional_res_df_lrg,pos_conditional_res_df_med,pos_conditional_res_df_shrt):
    f,ax = plt.subplots(1,3,figsize=(14,4),sharey=True)
    pos_conditional_res_df_lrg.boxplot(ax=ax[0])
    ax[0].set_title(r'$large-scale (threshold=%s)$'%thr,fontsize=20)
    ax[0].set_xlabel(r'$days$',fontsize=24)
    ax[0].set_ylabel(r'$\mu(r_t)$',fontsize=24)

    pos_conditional_res_df_med.boxplot(ax=ax[1])
    ax[1].set_title(r'$medium-scale$',fontsize=20)
    ax[1].set_xlabel(r'$days$',fontsize=24)

    pos_conditional_res_df_shrt.boxplot(ax=ax[2])
    ax[2].set_title(r'$short-scale$',fontsize=20)
    ax[2].set_xlabel(r'$days$',fontsize=24)
    plt.tight_layout()

################################################################
def plotLpplsOverSevT1AndKmeanTcVsDt(t2, data, res, filter=True, nCl=None, ax=None):
    
    """ Added on Okt.2016
        - Esimate LPPLS over several t1 for given t2 and
        - display the tc vs. t1 dependency using k-mean algo
    """
    
    from sklearn.cluster import KMeans
    #import seaborn as sns
    #sns.set_style('whitegrid')
    #sns.set_context('poster')
      
    # Munge results (filter)
    if filter == True:
        test = filter_big_df(res)
    else:
        test = res[res['w']>6]
        test = test[test['w']<13]
        #test = test[test['SSE']<1]
        #test = test[test['m'] > 0.]
        #test = test[test['m'] < 1.]
        test = test[test['minimum'] == 1]
        #test = test[test['sloppyness'] < 5.]
        #test = test[test['tc'] > 0.]
        test = test[test['tc'] < 800.]
        test = test[test['tc'] > -60.]
        #test = test[test['tc'] > 0.]
        #test = test[test['B'] < 0.]

    try:
        # nu. clusters
        if nCl == None:
            X = test[['tc', 'dt']]
            nCl = getOptimalKValue(X, getVec=False)
        else:
            X = test[['tc', 'dt']]
            nCl = nCl
        
        # K-mean
        est = KMeans(n_clusters=nCl)
        est.fit(X)
        all_labels = np.unique(est.labels_)
    
        # MeanTC -> mean value of clusters:
        meanTC = np.ones(nCl)
        for lb in all_labels:
            meanTC[lb] = int(np.mean(X.tc[est.labels_==lb]))
    except:
        pass
    
    if ax == None:
        
        # Lower bound for the plot
        lb = pd.Timestamp(t2) - 800 * pd.Timedelta('1D')
        ub = pd.Timestamp(t2) + 250 * pd.Timedelta('1D')
        
        import seaborn as sns
        sns.set_style('whitegrid')
        
        # Plot
        days = 250
        f, ax = plt.subplots(1,1,figsize=(12,7))
        data.plot(ax=ax, color='k', linewidth=3)
        ax.set_ylabel(r'$\ln[P(t)]$',fontsize=20)
        plt.hold()
        for i in range(len(test)):
            ax.hold(True)
            fit  = lp.lppl_project(test.ix[i], np.log(data), days=days)
            fit['fit'].plot(ax=ax, linewidth=2.)
            fit['projection'].plot(ax=ax, linewidth=2.)
            ax.legend('')
            ax.axvline(t2, color='k', linewidth=3)
            ax.set_title('Qualified Fits (frac.) = %.2f at t2 = %s'%(len(test)/np.float(len(res)),t2), fontsize=16)
            #fit['P'].plot()
            ax.set_xlim([str(lb)[0:10],str(ub)[0:10]])
            ax.set_ylim([data[str(lb)[0:10]:str(ub)[0:10]].min()[0],(data[str(lb)[0:10]:str(ub)[0:10]].max()[0]+1.)])


        
        #ax.set_ylim([2.5, 4.5])
        
        try:
            # KMEAN
            a = plt.axes([0.17, 0.59, .35, .35], axisbg='w')
            a.set_ylabel(r'$t_1$',fontsize=18)
            sns.kdeplot(X[['tc','dt']],shade=False, ax=a, cmap='bone', zorder=1, bw=(15,15))
            COLORS = 'rbkmgcy'*10
            for lb, cl in zip(all_labels, COLORS):
                a.scatter(X.tc[est.labels_==lb],X.dt[est.labels_==lb], marker='o',
                          color=cl,s=55,label='clstr %s: $N:$ %s $\mu$: %.0f $\sigma$: %s'%(lb,int(np.size(X.tc[est.labels_==lb])),
                                                                                            int(np.mean(X.tc[est.labels_==lb])),
                                                                                            int(np.std(X.tc[est.labels_==lb]))))
            lb=a.get_xticks().tolist()
            a.axvline(0, color='k',linestyle='--')
            dateVec = [pd.Timestamp(t2) + lb[i] * pd.Timedelta('1D') for i in range(len(lb))]
            V = [str(dateVec[i])[0:10] for i in range(len(dateVec))]
            a.set_xticklabels(V, rotation=50)
            a.legend(loc='lower left', fontsize=12)
            a.set_ylim(0, 800)
            plt.tight_layout()
        except:
            pass
                #if exception is met we dont plot the clusters
           
    
    else:
        # Lower bound for the plot
        lb = pd.Timestamp(t2) - 900 * pd.Timedelta('1D')
        ub = pd.Timestamp(t2) + 250 * pd.Timedelta('1D')
        
        days = 250
        data.plot(ax=ax, color='k', linewidth=1.5)
        ax.set_ylabel(r'$\ln[P(t)]$',fontsize=20)
        ax.set_title('Qualified Fits (frac.) = %.2f p.p. at t2 = %s'%(len(test)/np.float(len(res)),t2), fontsize=16)

        # Iterate
        for i in range(len(test)):
            ax.hold(True)
            fit  = lp.lppl_project(test.ix[i], data, days=days)
            fit['fit'].plot(ax=ax, linewidth=1.2, alpha=0.7)
            fit['projection'].plot(ax=ax, linewidth=.7, linestyle='-')
            ax.legend('')
            ax.axvline(t2, color='k', linewidth=3, linestyle='--')
            #ax.set_xlim(['2014-04-01','2017-06-01'])
            ax.set_xlim([str(lb)[0:10],str(ub)[0:10]])
            ax.set_ylim([data[str(lb)[0:10]:str(ub)[0:10]].min()[0],(data[str(lb)[0:10]:str(ub)[0:10]].max()[0]+1.)])
        
        ax.legend('')
        plt.tight_layout()

        # K-mean
        #est = KMeans(n_clusters=3)
        #X = test[['tc','dt']]
        #est.fit(X)
        #all_labels = np.unique(est.labels_)

        # MeanTC -> mean value of clusters:
        #meanTC = np.ones(3)
        #for lb in all_labels:
        #    meanTC[lb] = int(np.mean(X.tc[est.labels_==lb]))


        # KMEAN
        #a = plt.axes([0.16, 0.57, .35, .35], axisbg='w')
        #a.set_ylabel(r'$t_1$',fontsize=18)
        #sns.kdeplot(X[['tc','dt']],shade=False, ax=a, cmap='bone', zorder=1, bw=(15,15))
        #COLORS = 'rbkmgcy'*10
        #for lb, cl in zip(all_labels, COLORS):
        #    a.scatter(X.tc[est.labels_==lb],X.dt[est.labels_==lb], marker='o',
        #              color=cl,s=55,label='clstr %s: $N:$ %s $\mu$: %.0f $\sigma$: %s'%(lb,int(np.size(X.tc[est.labels_==lb])),
        #                                                                                int(np.mean(X.tc[est.labels_==lb])),
        #                                                                                int(np.std(X.tc[est.labels_==lb]))))
        #lb=a.get_xticks().tolist()
        #a.axvline(0, color='k',linestyle='--')
        #dateVec = [pd.Timestamp(t2) + lb[i] * pd.Timedelta('1D') for i in range(len(lb))]
        #V = [str(dateVec[i])[0:10] for i in range(len(dateVec))]
        #a.set_xticklabels(V, rotation=50)
        #a.legend(loc='lower right', fontsize=12)
        #plt.tight_layout()

################################################################
"""
Jan @ 2017
Here there is the double plot (contour plot steeming from the MPL method) and the LPPLS fits (above)
Developed for the bitcoin paper
"""

####################################
def calculateEverythingForGivenT2(data, t2):
    # Data must be a vector dataframe
    
    # Fits LPPLS at a given t2
    parsDF = run_search_t1_paralel(data, t2, fit_ABC=True, job_pool=None)
    
    # Get Contour using the MPL
    allRes, good = lpi.createContourPlotUsingModLikForGivenT2(data, t2)
    
    return parsDF, allRes, good


####################################
def fullPlot(data, parsDF, allRes, good, t2):
    # Initialize
    f,ax = plt.subplots(1,2,figsize=(12,5))
    axs = ax.ravel()
    
    # Plot left
    plotLpplsOverSevT1AndKmeanTcVsDt(t2, data, parsDF, filter=False, ax=axs[0])
    
    #plot right
    lpi.plotContour(allRes, good, ax=axs[1])
    plt.tight_layout()


####################################
def finalFunc4Movies(data, t2):
    
    """ Function that fits data at t2 for several t1 using the LPPLS and MPL methods and plot results 
        For creating pdf movies use:
    
            import matplotlib.backends.backend_pdf
            
            ## HERE WEPLOT THE MOVIE
            start =  pd.datetime(2013, 1, 2)
            end   = pd.datetime(2014, 6, 1)
        
            # T2 range
            rng   = pd.date_range(start, end,freq='M'); rng=rng.values.astype('M8[D]')
        
            # Set data
            Y = pd.DataFrame(data[data.columns[-1]])
        
            # Name outputed PDF
            pdf = matplotlib.backends.backend_pdf.PdfPages("/Users/demos/Desktop/bubble1.pdf")
            for v in range(len(rng)):
            finalFunc4Movies(Y, str(rng[v]))
            pdf.savefig()
            pdf.close()
    """
    
    # Calculate
    parsDF, allRes, good = calculateEverythingForGivenT2(data, t2)
    
    # Plot
    fullPlot(data, parsDF, allRes, good, t2)


####################################
def ApplyKmeanAndPlot(data, nCl = 3, normalize=True):

    """
    Compute K-Mean and plot : Jan@17
    """

    # Require the following packages
    from sklearn.cluster import KMeans
    import portfolio_functions as pf

    # Normalize data using the pf functions
    if normalize == True:
        data = pf.normalize_data_for_comparacy(data, sklearn=True)
    else:
        pass

    # Compute K-Mean
    est = KMeans(n_clusters=nCl)
    X = data[[data.columns[0], data.columns[1]]]
    est.fit(X)
    all_labels = np.unique(est.labels_)

    # Plot
    COLORS = 'rbkmgcy' * 10
    f, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    for lb, cl in zip(all_labels, COLORS):
        plt.scatter(data[data.columns[0]][est.labels_ == lb],
                    data[data.columns[1]][est.labels_ == lb],
                    marker='o', color=cl, s=55, facecolor='w')

    plt.tight_layout()


####################################
def getOptimalKValue(data, getVec=False):
    """ Return the optimal value of K for a given [[1,1]] data set.
        from sklearn.cluster import KMeans
        For getting the vector for ploting the elbow methdod -> True
        else, return scalar with the K*.
        -> Get the value that maximies the silhouete score.
    """

    # Define range of clusters
    range_n_clusters = [2, 3, 4, 5, 6]
    S = []

    for n_clusters in range_n_clusters:
        est = KMeans(n_clusters=n_clusters)
        cluster_labels = est.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        S.append(silhouette_avg)

    Kstar = pd.DataFrame(S, index=range_n_clusters)
    kStarScalar = Kstar.nlargest(1, 0).index[0]

    if getVec == False:
        return kStarScalar
    else:
        return Kstar