#################################################################################
# Library for data handling for DS-DS project

# - we have the Hierarchy bubble indicator constructor here
# - The BStatus indicator is also here
#################################################################################
# (c) Guilherme Demos, 2015
# gdemos@ethz.ch
#################################################################################
import pandas as pd
import numpy as np
import os
import datetime as dt
import re
import time
import matplotlib.pyplot as plt
import h5py
import sqlalchemy
from pandas.io import sql
from os.path import join
import os
import statsmodels.api as sm
import sys

try:
    from numba import jit
except:
    def jit(func):
        return func

from pylppl_model import lppl

# Define paths
path_HD = '/Volumes/Demos BackupHD/Signals/0/'
path_bcktst_files = '/Users/demos/Desktop/BACKTESTING/'
path2 = '/Users/demos/bigdata/data_base/'
sys.path.append('/Users/demos/Documents/Python/ipy (work)/LPPLS - Sloppy/')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/Portfolio optimization - bt package')
import portfolio_functions as pf
import sloppy_func as fsl

import matplotlib as mp

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 22}

label_size = 14
mp.rcParams['xtick.labelsize'] = label_size
mp.rcParams['ytick.labelsize'] = label_size

#################################################################################
# paths definition
#################################################################################
def get_idx_from_benchmark(path2):
    try:
        # List the entire contents of the file
        f = h5py.File(path2 + "SPCOMP.h5")
        list_of_names = []
        f.visit(list_of_names.append)
    except:
        path2 = '/Users/demos/bigdata/data_base/'
        f = h5py.File(path2 + "SPCOMP.h5")
        list_of_names = []
        f.visit(list_of_names.append)

    # Get SP500 constituent stocks
    dataset = f['ids']
    IDX = pd.Series(dataset.value)  # Index Containing Assets

    return IDX

#################################################################################
# Filtering for data
#################################################################################
def get_data_frame_for_ID(ID, paths, path_HD, convert_dates=True):
    """ Function that runs it all """

    # Return the path
    paths = getFiles(path_HD)

    # Get ID and its corresponding location
    ID, Location = rtrn_id_and_fileLoc(paths, ID)

    # Return the id's within file 395.h5
    f, lst_keys = get_file_keys(ID, path_HD, Location)

    # IMPORTANT STEP: CHOOSE KEY id (automatically):
    idkey = [k for k in lst_keys if k == 'id' + str(ID)]

    # Get parameters for that specific id
    results = get_dataframe_from_key(idkey[0], f, convert_dates=convert_dates)

    return results

#################################################################################
def get_file_keys(fname, path_HD, Location):
    """ Return list of keys """

    lst_keys = []

    f = h5py.File(Location, 'r')

    keys = f.keys()
    for key in keys:
        print('File: %s -> %s' % (fname, key))
        lst_keys.append(key)
    # f.close()

    # If debug:
    # print('numb. keys = %s. Choose the one I want'%np.size(lst_keys))

    return f, lst_keys

#################################################################################
def get_dataframe_from_key(idkey, f, convert_dates=False):
    """
    This function returns in a DF format all the lppls 
    fits of the asset [idx] -> idkey (_string).
    """

    # lbls
    lbls = ['aid', '_t2', '_t1', '_A', '_B', '_C', '_m', '_w', '_phi',
            '_tc', '_tuning', '_nosc', '_implieddd', '_sse']

    # F() to transform dates
    to_dt = lambda x: dt.datetime.fromordinal(int(x)) + dt.timedelta(days=x % 1)

    # Crucial step
    ts_inside_key = list(f[idkey])  # time dates.h5 which contains parameters values

    # Append everything into a DF
    D = []

    for T2 in ts_inside_key:
        k1, k2 = np.shape(f[idkey + '/' + T2])
        for lenFile in range(k1):
            D.append(f[idkey + '/' + T2][lenFile])  # [Nx14] parameter values # HERE WAS THE PROBLEMM !

    # Create DF of results
    results = pd.DataFrame(D, columns=lbls)  # Make dataframe

    # if one wants transformed into python data
    if convert_dates == True:
        results['_t2'] = results['_t2'].apply(to_dt)
        results['_t1'] = results['_t1'].apply(to_dt)
        results['_tc'] = results['_tc'].apply(to_dt)

        return results[results._sse > 0]
    else:
        return results[results._sse > 0]

    return results[results._sse > 0]

#################################################################################
def getFiles(path):
    '''Gets all of the files in a directory
       Path of the files: is a dictionary with keys = xxx.h5 and items '....../xxx.h5'
       OK -
    '''
    sub = os.listdir(path)
    paths = {}
    for p in sub:
        # print p # just to see where the function has reached; it's optional
        pDir = os.path.join(path, p)
        if os.path.isdir(pDir):
            paths.update(getAllFiles(pDir, paths))
        else:
            paths[p] = pDir

    return paths

#################################################################################
def rtrn_id_and_fileLoc(paths, ID):
    # Get keys
    pKeys = paths.keys()
    pItem = paths.items()

    # matching => [paths.items() for s in pKeys if ID in s]
    try:
        Location = paths.get(ID)
    except:
        Location = paths.get(str(ID) + '.h5')

    if Location is None:
        F = get_file_loc(ID, paths)
        Location = paths.get(F[0][1])
    else:
        pass

    # print results
    print('[ID] => %s is in [Location] => %s' % (ID, Location))

    return ID, Location

#################################################################################
def save_results(res, fname, path):
    """ Save a given file to HDF format """
    res.to_hdf(path + fname + '.h5', 'res')

#################################################################################
# DATAFRAME FUNCTIONS:
#################################################################################
def get_prcTmeSrs_prmtrs(ID, data, paths, path_HD, convert_dates=True):
    """ return the price time series on a DF format (Y) and
        return estimated parameters for that time seris on a DF (results)
    """

    # obtain parameters as DF for a given ID
    results = get_data_frame_for_ID(ID, paths, path_HD, convert_dates=convert_dates)

    # Create DF from data
    Y = pd.DataFrame(data.ix[ID].close);
    Y.index = data.ix[ID].time

    return results, Y


#################################################################################
def Q_filter(fits):
    """ Filter a given data frame containing LPPLS fits (from qunzhi)
        TO BE USED WITH BACKTESTING DATAFRAMES
    """
    inds = ((fits['_m'] > 0.1) & (fits['_m'] < 0.89) &
            (fits['_w'] > 2) & (fits['_w'] < 25) &
            (fits['_tuning'] > 0.8) & (fits['_tuning'] < 1.2) &
            (fits['_sse'] < 0.05) & fits['_nosc'] > 1)
    fits = fits.copy()
    fits['pass'] = inds

    return fits

#################################################################################
def Q_filter_myW(fits, sharp_lims=True, pos_only=True, bi = True):
    """ Filter a given data frame containing LPPLS fits (from qunzhi)
        TO BE USED WITH BACKTESTING DATAFRAMES
        - If BI = True, desconsidero + or - bubbles and filter only the other parameters
    """

    if sharp_lims is not False and pos_only is not False and bi is not True:
        fits = fits[(fits['_m'] > 0.1) & (fits['_m'] < 0.9) &
                    (fits['_w'] > 6) & (fits['_m'] < 13) &
                    (fits['_tuning'] > 0.3) & (fits['_tuning'] < 33) &
                    (fits['_B'] < 0) & (fits['_nosc'] >= 1)
                    ]
        return fits
    elif sharp_lims is not True and pos_only is not False and bi is not True:
        fits = fits[(fits['_m'] > 0.1) & (fits['_m'] < 0.9) &
                    (fits['_w'] > 5) & (fits['_m'] < 16) &
                    (fits['_tuning'] > 0.1) & (fits['_tuning'] < 3) &
                    (fits['_B'] < 0) & (fits['_nosc'] >= 1)
                    ]
        return fits
    elif sharp_lims is not False and pos_only is not True and bi is not True:
        fits = fits[(fits['_m'] > 0.1) & (fits['_m'] < 0.9) &
                    (fits['_w'] > 6) & (fits['_m'] < 13) &
                    (fits['_tuning'] > 0.3) & (fits['_tuning'] < 33) &
                    (fits['_B'] > 0) & (fits['_nosc'] >= 1)
                    ]
        return fits
    elif bi == True:
        fits = fits[(fits['_m'] > 0.1) & (fits['_m'] < 0.9) &
                    (fits['_w'] > 6) & (fits['_m'] < 13) &
                    (fits['_tuning'] > 0.3) & (fits['_tuning'] < 33) &
                    (fits['_nosc'] >= 1)
                    ]
        return fits

#################################################################################    
def retrn_t1(params):
    """ return t1 for a given t2 at the time interval
        required for constructing the bubble scope 
    """

    t2Vec = [x for x in params['_t2'][:]]

    BD = [28, 33, 38, 43, 48]
    t2V = []
    t1V = []

    for t2 in range(len(params._t2)):
        for deltaT in BD:
            T1 = t2Vec[t2] - deltaT * pd.Timedelta('1D')
            if np.shape(params[params._t1 == T1]) == (0, 14):
                T1 = t2Vec[t2] - (deltaT - 1) * pd.Timedelta('1D')
            elif np.shape(params[params._t1 == T1]) == (0, 14):
                T1 = t2Vec[t2] - (deltaT - 2) * pd.Timedelta('1D')
            elif np.shape(params[params._t1 == T1]) == (0, 14):
                T1 = t2Vec[t2] - (deltaT - 3) * pd.Timedelta('1D')
            elif np.shape(params[params._t1 == T1]) == (0, 14):
                T1 = t2Vec[t2] - (deltaT - 4) * pd.Timedelta('1D')

            t1V.append(T1)

    return t1V

#################################################################################
def get_boolean_at_each_t1(params, t1V):
    """ Here we check if the "previous 5 t's" meet conditions
        then, we return 1 if True or Zero otherwise
        BASICALLY: Run this f() for each t2
    """

    BStatus_at_t2_vec = []
    BStatus_at_t2 = []

    for i in t1V:
        # condition := Returns the index where parameter t1 = [X]
        condition = np.shape(params.ix[params._t1 == i]) == (0, 14)
        if condition is not True:
            BStatus_at_t2_vec.append(1)
        else:
            BStatus_at_t2_vec.append(0)

    return BStatus_at_t2_vec

#################################################################################
def ret_boolean_at_5_t1(fv_bool_t1, t1V):
    """ Input: vector of 5*size(params._t1) - Boolean -
        For each five, condense into 1 or 0
        Hence, spits V = boolean(size(params._t1))
    """

    mult_cond = np.arange(0, len(t1V), 5)
    count = 0
    v = []
    t2V = []

    for i in range(len(fv_bool_t1)):
        v.append(fv_bool_t1[i])
        count += 1
        if count in mult_cond:
            if sum(v) == 5:
                t2V.append(1)
            else:
                t2V.append(0)
            v = []

    return t2V

#################################################################################
def run_full_bubbleIndConstruction_asset(params, price, disp_fig=True):
    """ Here we return in DF format: index=t2 and column[0] bubble status """

    # run initial func()
    t1V = retrn_t1(params)

    # function() 1 
    BStatus_at_t2 = get_boolean_at_each_t1(params, t1V)

    # function() 2
    t2V = ret_boolean_at_5_t1(BStatus_at_t2, t1V)

    # plot
    DFBS = pd.DataFrame(t2V)
    DFBS.index = params._t2[0:-1]

    if disp_fig is not False:
        DFBS.plot(color='r', legend=False)
        ax = plt.twinx()
        price.plot(ax=ax, legend=False)
        plt.tight_layout()

    return DFBS

#################################################################################
def get_assetID(ID, data):
    """
    :param ID: ID of asset within the full dataframe of assets [data]
    :param data: [full dataframe containing all assets]
    :return: DF with values of the requested ID
    """

    Y = data.ix[ID]
    Y.index = Y.time
    Y.drop('time', axis=1)

    return Y

#################################################################################
def rtrn_indicator(GOOD):

    """ Return a dataFrame with the simple criteria:
        - If the previous 5 t1's of a given t2 pass our filter criteria,
           we are in a bubble and for that t2, our DF = 1.
        - If thats not the case, DF = 0
    """

    # Make t2 the index
    GOOD.reset_index()
    GOOD.index = GOOD._t2

    # Get unique t2's
    t2V = GOOD['_t2'].unique()

    # Bubble index at t2
    BI_atT2 = []

    for t2 in range(len(t2V)):
        T2 = t2V[t2]
        if np.shape(GOOD.ix[T2]) >= (5,14) and (GOOD.ix[T2]._B < 0).all():
            BI_atT2.append(1)
        elif np.shape(GOOD.ix[T2]) >= (5,14) and (GOOD.ix[T2]._B > 0).all():
            BI_atT2.append(-1)
        else:
            BI_atT2.append(0)

    # Make dataframe
    BI_DF = pd.DataFrame(BI_atT2,columns=[int(GOOD.aid.values[1])],index=[t2V])

    return BI_DF

#################################################################################
def plot_Bindicator(Y,BI):
    f,ax = plt.subplots(1,1,figsize=(8,4), sharex=True)
    Y.close.plot(lw=0.7,color='black',ax=ax)
    [ax.axvline(BI.index[s], color='red', linestyle='--',
                   lw=0.6, alpha=0.6) for s in range(len(BI.index))]
    plt.tight_layout()

#################################################################################
def Filter_and_create_bubbleInd_asset(params, ID, data,
                                      plot_BI=False, sharp_lims=True, pos_only=True, bi=True, save_hdf=False):

    """
    :param params: estimated parameters (dirty)
    :param ID: ID of the asset I want
    :param data: time series dataframe as vladimirs
    :param plot_BI: Plot B_status_indicator
    :param sharp_lims: Use sharp limits for filtering?
    :param pos_only: Positive bubbles only?
    :return:
    """

    ## Filter
    GOOD = Q_filter_myW(params,
                            sharp_lims=sharp_lims, pos_only=pos_only, bi=bi)

    ## Calculate the indicator
    BI = rtrn_indicator(GOOD)

    if plot_BI is not False:
    ## get ID dataFrame of price
        Y = get_assetID(ID, data)

        ## Plot Bubble indicator
        plot_Bindicator(Y,BI)

    if save_hdf is not False:
        save_results(BI,'BI_'+str(ID),'/Volumes/Demos BackupHD/Signals/')

    return BI

#################################################################################
def BSIndicator(ID, data, paths, plot_BI=False, sharp_lims=True, pos_only=True, bi=False, save_hdf=True):

    """

    :param ID: asset
    :param data: price time seris
    :param paths: paths for finding the .h5 file [ID.h5] |dfn.getFiles(path_HD)|
    :param plot_BI: plot bubble indicator ?
    :param sharp_lims: sharp limits on filtering ?
    :param pos_only: positive only ?
    :param save_hdf: Save on hdf5 ?
    :return: params data frame (unfiltered), price dataframe and BI_DF
    """

    # TO PEGANDO DUAS VEZES A SERIE DE TEMPO!! OTIMIZAR ISSO PO!

    # | get parameters and time series |
    params, price = get_prcTmeSrs_prmtrs(ID, data, paths, path_HD, convert_dates=True)

    # | Run the bubble indicator constructor |
    BI_DF = Filter_and_create_bubbleInd_asset(params, ID, data,
                                              plot_BI = plot_BI, sharp_lims = sharp_lims,
                                              pos_only = pos_only, bi=bi, save_hdf = save_hdf)

    return params, price, BI_DF

#################################################################################
def get_file_loc(ID, paths):

    """ Search for file ID: return [ID,loc.h5]
        When the asset is on a different .h5 file
    """

    # append file
    F = []

    # directories to search
    H5_search = [paths.items()[s] for s in range(len(paths.items()))]

    # read
    for i in range(len(paths.items())):
        try:
            f = h5py.File(H5_search[i][1], 'r')

            B = [str('id' + str(ID)) in f.keys()]
            if B == [True]:
                F.append([str(ID), H5_search[i][0]])
            else:
                pass
        except:
            pass

    return F

#################################################################################
# NOT GOOD
def plot_bubble_indicator(Y, BI, pos_only=False,
                          neg_only=False, save=False):

    cols = BI.columns.values[0]

    f,ax = plt.subplots(1,1,figsize=(14,6))
    ax.fill_between(Y.index,Y.close,alpha=0.1)
    ax.plot(Y.index,Y.close,'k')
    plt.title('ID: %s' %cols)
    ax.set_ylabel('p(t)')
    ax1=ax.twinx()
    ax1.plot(Y.index,Y.volume,alpha=0.1)
    ax1.set_ylabel('volume')

    if pos_only is not False and neg_only is not True:
        [ax.axvline(BI.index[s], color='red', linestyle='--',
                    lw=0.6, alpha=0.6) for s in range(len(BI.index)) if BI.values[s] == 1]
    if pos_only is not True and neg_only is not False:
        [ax.axvline(BI.index[s], color='green', linestyle='--',
                    lw=0.6, alpha=0.6) for s in range(len(BI.index)) if BI.values[s] == -1]
    if pos_only is not False and neg_only is not False:
        [ax.axvline(BI.index[s], color='red', linestyle='--',
                    lw=0.6, alpha=0.6) for s in range(len(BI.index)) if BI.values[s] == 1]
        [ax.axvline(BI.index[s], color='green', linestyle='--',
                    lw=0.6, alpha=0.6) for s in range(len(BI.index)) if BI.values[s] == -1]
    plt.tight_layout()

    if save is not False:
        try:
            plt.savefig('/Users/demos/Desktop/test.pdf')
        except:
            print('choose proper path')

#################################################################################
def plot_bubble_indicator_2(Y, BI, pos_only=False,
                          neg_only=False, save=False):

    cols = BI.columns.values[0]

    f,ax = plt.subplots(1,1,figsize=(14,6))
    [ax.axvline(BI.index[s], color='red', linestyle='--',
                    lw=0.6, alpha=0.6) for s in range(len(BI.index)) if BI.values[s] == 1]
    [ax.axvline(BI.index[s], color='green', linestyle='--',
                    lw=0.6, alpha=0.6) for s in range(len(BI.index)) if BI.values[s] == -1]
    ax.plot(Y.index,Y.close,'k')
    plt.title(r'$ID$: %s' %cols)
    ax.set_ylabel(r'$p(t)$')



    if save is not False:
        try:
            plt.savefig('/Users/demos/Desktop/test.pdf')
        except:
            print('choose proper path')

####################################################################################
def simple_plot(ID, data, loc, pos_only=True, neg_only=True, save=False):

    """ enter ID and path, plot all  """

    Y = get_assetID(ID, data)

    ID = str(ID)
    BI = pd.read_hdf(loc+'BI_'+ID+'.h5','res')

    plot_bubble_indicator_2(Y, BI, pos_only=pos_only,
                          neg_only=neg_only, save=save)

    return Y, BI

####################################################################################
# Create the bubble status out of the dataframes created "BI_[ID].h5"
####################################################################################
def get_true_t2_range_from_index(path2='/Users/demos/bigdata/'):
    try:
        # List the entire contents of the file
        f = h5py.File(path2 + "SPCOMP.h5")
        list_of_names = []
        f.visit(list_of_names.append)
    except:
        path2 = '/Users/demos/bigdata/'
        f = h5py.File(path2 + "SPCOMP.h5")
        list_of_names = []
        f.visit(list_of_names.append)

    # Get SP500 time deltas
    dataset = f['times']

    # create a dataframe where the index = all t2's that exist
    DATAS = pd.DataFrame(dataset.value,columns=['_t2'])

    # F() to transform dates
    to_dt = lambda x: dt.datetime.fromordinal(int(x)) + dt.timedelta(days=x % 1)

    # convert
    DATAS ['_t2'] = DATAS['_t2'].apply(to_dt)
    DATAS.index = DATAS._t2

    return DATAS

####################################################################################
def compound_BI(FILES, path2, loc, fillNA=False, summ=False):
    
    """ read all the BI_IDS calculated on brutus and return the big (Nx1) matrix
        path2    = location where the sp500 is
        path_BSI = path where the bsindicator is
    """
    
    # Get true range t2 dataframe
    if path2 is None:
        T2_index = get_true_t2_range_from_index(path2='/Users/demos/bigdata/')
    else:
        T2_index = get_true_t2_range_from_index(path2)
    T2_index.drop('_t2', axis=1, inplace=True)
    
    # concatenate
    for I_D in range(len(FILES)-1):
        try:
            DF = pd.read_hdf(loc+FILES[1+I_D],'res')
            ID_col = DF.columns; ID_col = ID_col.values[0]

            T2_index = pd.concat([T2_index, DF], axis=1)
        except:
            pass
        
    # clean the NaN values from the BIGMATRIX
    # Fill NaN values with 0
    if fillNA is not False:
        T2_index = T2_index.fillna(0)
        if summ is not False:
            BINDICATOR = [np.sum(T2_index.ix[s].values,dtype=np.float64) for s in T2_index.index]
            val = np.array(BINDICATOR)
        else:
            BINDICATOR = [np.mean(T2_index.ix[s].values,dtype=np.float64) for s in T2_index.index]
            val = np.array(BINDICATOR)
    if fillNA is not True:
        ## OLD
        BINDICATOR = [T2_index.ix[s].mean() for s in T2_index.index]
        val = pd.DataFrame(BINDICATOR,columns=['mbsi'])
        #BINDICATOR = [np.sum(T2_index.ix[T2])/(np.shape(T2_index.ix[T2][T2_index.ix[T2] == 0])[0] + abs(np.sum(T2_index.ix[T2]))) for T2 in T2_index.index]
        #val = BINDICATOR
        
    return T2_index, val

####################################################################################
def plot_MBSI_from_big_matrix(Y,T2_index,val,trend=None):
    
    """ T2_index -> big matrix dataframe
        Y -> [t2,Y] dataframe
        # import statsmodels.api as sm
        # cycle, trend = sm.tsa.filters.hpfilter(val, lamb=1600)
    """
    
    nrows = 3

    plt.figure(figsize=(11, 6))
    ax  = plt.subplot2grid((nrows, 1), (0, 0),
                          colspan=1, rowspan=nrows-1)
    axx = plt.subplot2grid((nrows, 1), (nrows-1, 0),
                           colspan=1, rowspan=1, sharex=ax)
    ax2 = ax.twinx()

    if trend is None:
        #
        ax.plot(Y.index,np.log(Y.values),color='black',lw=0.7)
        ax2.plot(T2_index.index,val,color='r',lw=0.5,alpha=0.5)
        ax2.axhline(0,color='black',linestyle='--')
        #
        axx.plot(T2_index.index,val,color='r',lw=0.3,alpha=0.9)
        axx.axhline(0,color='black',linestyle='--')
        #
        axx.set_ylabel(r'$B. S. Indicator$',fontsize=18)
        ax.set_ylabel(r'$\ln[S&P\,500$]',fontsize=18)
        plt.tight_layout()
    else:
        #
        ax.plot(Y.index,np.log(Y.values),color='black',lw=0.7)
        #
        axx.plot(T2_index.index,val,color='b',lw=0.5,alpha=0.4)
        axx.plot(T2_index.index,trend,'k',lw=1)
        axx.axhline(0,color='black',linestyle='--')
        axx.axhline(-np.std(trend),color='black',linestyle='--',lw=1)
        axx.axhline(np.std(trend),color='black',linestyle='--',lw=1)
        #
        axx.set_ylabel(r'$B. S. Indicator$',fontsize=18)
        ax.set_ylabel(r'$\ln[S&P\,500$]',fontsize=18)
        #
        a = plt.axes([0.19, 0.65, .26, .3], axisbg='w')
        a.set_ylabel(r'$pdf(BSI)$')
        a.hist(trend,bins=90,color='blue',alpha=0.5)
        a.axvline(0,lw=4,color='red',linestyle='--')
        a.axvline(-np.std(trend),color='red',linestyle='--',lw=2)
        a.axvline(np.std(trend),color='red',linestyle='--',lw=2)

        plt.tight_layout()

####################################################################################
def plot_MBSI_zoom(MBI_vec, Y, init, last, trend, retAX=False):

    #HP_F_DF = pd.DataFrame(trend, columns=['hpfltr'])
    #HP_F_DF.index = MBI_vec.index.copy()
    MBI_vec['hpfltr'] = trend
    
    if retAX is not False:
        Y[init:last].plot(color='blue', lw=0.9,legend=False,ax=retAX)
        ax2 = retAX.twinx()
        #MBI_vec[init:last].plot(color='blue', lw=0.2, ax=ax2, legend=False)
        MBI_vec[init:last].hpfltr.plot(color='red', lw=0.7, ax=ax2, legend=False)
        ax2.axhline(np.mean(MBI_vec[init:last].values),
                    color='k', linestyle='--', lw=2)
        ax2.axhspan(-np.std(MBI_vec[init:last].values),
                    np.std(MBI_vec[init:last].values),
                    color='red', alpha=0.15)
        ax2.axhline(0,color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel(''); retAX.set_xlabel('');
        retAX.set_ylabel(r'$\ln[P(t)]$'); ax2.set_ylabel(r'$BSIndicator$');
        plt.tight_layout()
    else:
        f,ax = plt.subplots(1,1,figsize=(8,4))
        Y[init:last].plot(color='blue',legend=False,ax=ax)
        ax.set_xlabel('')
        ax2 = ax.twinx()
        #MBI_vec[init:last].plot(color='blue', lw=0.2, ax=ax2, legend=False)
        MBI_vec[init:last].hpfltr.plot(color='red', lw=1.5, ax=ax2, legend=False)
        ax2.axhline(np.mean(MBI_vec[init:last].values),
                    color='k', linestyle='--', lw=2)
        ax2.axhspan(-np.std(MBI_vec[init:last].values),
                    np.std(MBI_vec[init:last].values),
                    color='red', alpha=0.15)
        ax2.axhline(0,color='black',linewidth=1)
        ax2.set_xlabel('')
        plt.tight_layout()

####################################################################################
def plt_bsi(T2_index, val, Y, init, last, hp_filter = True, show_mean = False, retAX=False):

    #V = val.copy()
    # transform val into a DF
    V = pd.DataFrame(val,columns=['val'])
    V.index = T2_index.index

    if hp_filter == True:
        cycle, trend = sm.tsa.filters.hpfilter(val, lamb=2000)
        V['trend'] = trend

    if retAX is not False:
    # start plotting
        Y[init:last].plot(color='black', lw=1.3,legend=False,ax=retAX)
        ax2 = retAX.twinx()
        V[init:last].val.plot(color='blue', lw=1.3, ax=ax2, alpha=0.3, legend=False)
        #V[init:last][V[init:last]['val'] >= 0.01].plot(color='red', lw=0.9, ax=ax2, alpha=0.3, legend=False)
        #V[init:last][V[init:last]['val'] <= 0.01].plot(color='green', lw=0.9, ax=ax2, alpha=0.3, legend=False)
        try:
            V[init:last].trend.plot(color='red', lw=1.5, ax=ax2, legend=False)
        except:
            pass
        ax2.axhline(np.mean(V[init:last].val.values),color='black',linestyle='--',linewidth=1.2)
        ax2.axhline(0,color='black',linestyle='-',linewidth=1.5)
        #ax2.axhline(0,np.mean(V[init:last].val.values),color='black',linestyle='--',linewidth=1.2)
        ax2.axhline(-np.std(V[init:last].val.values),color='black',linestyle=':',linewidth=1.5)
        ax2.axhline(np.std(V[init:last].val.values),color='black',linestyle=':',linewidth=1.5)
        if show_mean is not False:
            if np.mean(V[init:last].val.values) > 0:
                ax2.axhspan(0, np.mean(V[init:last].val.values),color='red',alpha=0.2,linewidth=1.2)
            else:
                ax2.axhspan(0, np.mean(V[init:last].val.values),color='green',alpha=0.2,linewidth=1.2)
        else:
            pass
        ax2.set_xlabel(''); retAX.set_xlabel('');
        retAX.set_ylabel(r'$\ln[P(t)]$'); ax2.set_ylabel(r'$BSIndicator$');
        plt.tight_layout()
    else:
        Y[init:last].plot(color='blue', lw=0.9,legend=False)
        ax = plt.twinx()
        V[init:last].plot(color='blue', lw=0.3, alpha=0.9, legend=False, ax=ax)
        plt.tight_layout()

####################################################################################
def plt_bsi_sector(T2_index, val, Y, init, last, sector_vec, hp_filter = True, retAX=False):

    """ plot for visualizing sectors
    """
    #V = val.copy()
    # transform val into a DF
    V = pd.DataFrame(val,columns=['val'])
    V.index = T2_index.index

    if hp_filter == True:
        cycle, trend = sm.tsa.filters.hpfilter(val, lamb=2000)
        V['trend'] = trend

    if retAX is not False:
    # start plotting
        Y[init:last].plot(color='black', lw=0.9,legend=False,ax=retAX)
        ax2 = retAX.twinx()
        V[init:last].val.plot(color='blue', lw=0.9, ax=ax2, alpha=0.2, legend=False)
        try:
            V[init:last].trend.plot(color='red', lw=0.9, ax=ax2, legend=False)
        except:
            pass
        ax2.axhline(np.mean(V[init:last].val.values),color='black',linestyle='--',linewidth=1.2)
        ax2.axhline(0,color='black',linestyle='-',linewidth=1.5)
        ax2.axhline(0,np.mean(V[init:last].val.values),color='black',linestyle='--',linewidth=1.2)
        ax2.axhline(-np.std(V[init:last].val.values),color='black',linestyle=':',linewidth=1.5)
        ax2.axhline(np.std(V[init:last].val.values),color='black',linestyle=':',linewidth=1.5)
        if np.mean(V[init:last].val.values) > 0:
            ax2.axhspan(0, np.mean(V[init:last].val.values),color='red',alpha=0.2,linewidth=1.2)
        else:
            ax2.axhspan(0, np.mean(V[init:last].val.values),color='green',alpha=0.2,linewidth=1.2)
        ax2.set_xlabel(''); retAX.set_xlabel('');
        retAX.set_ylabel(r'$\ln[P(t)]$'); ax2.set_ylabel(r'$BSIndicator$');
        retAX.set_title(r'$Sector: %s$'%sector_vec,fontsize=16)
        ax2.set_ylim([-1,1])
        plt.tight_layout()
    else:
        Y[init:last].plot(color='blue', lw=0.9,legend=False)
        ax = plt.twinx()
        V[init:last].plot(color='blue', lw=0.3, alpha=0.5, legend=False, ax=ax)
        plt.tight_layout()

####################################################################################
# Sector construction
####################################################################################
def create_key_sector_dataframe(IDX):
    # read 'asset_sectors'
    
    f = h5py.File(path2+"SPCOMP_sig_00.h5")
    list_of_names = []
    f.visit(list_of_names.append)

    dataset = f['asset_sectors']
    dataset = pd.Series(dataset)

    SECTOR_DATAFRAME  =  pd.DataFrame(IDX)
    SECTOR_DATAFRAME['sector'] = dataset
    SECTOR_DATAFRAME.head()
    
    return SECTOR_DATAFRAME

####################################################################################
def retrn_sec_for_IDX(myID, S_IDX):
    """ search for IDX that I have on IDX_FULL and return the corresponding sector
        associated with that asset (single one)
    """
    # retorno o sector para a myID
    # S_IDX[0] -> Vetor contendo todas as IDS possiveis (1075x2)
    #          -> col[0]: assets and col[1]: sectors
    # Aqui, vemos se myID esta contido em S_IDX[0]. Se sim, retorna o .ix da DF com sua locali
    # zacao. .sector.values[0] me da o corresponding sector desse ativo.

    myID_sctr = S_IDX.ix[S_IDX[0]==myID].sector.values[0]

    return myID_sctr

####################################################################################
def retrn_sec_for_IDX_all(S_IDX,S_MYIDX):
    """ search for IDX that I have on IDX_FULL and return the corresponding sector
        associated with that asset
    """
    # retorno o sector para a myID
    # S_IDX[0] -> Vetor contendo todas as IDS possiveis (1075x2)
    #          -> col[0]: assets and col[1]: sectors
    # Aqui, vemos se myID esta contido em S_IDX[0]. Se sim, retorna o .ix da DF com sua locali
    # zacao. .sector.values[0] me da o corresponding sector desse ativo.

    myID_sctr = []

    for IDD in range(len(S_MYIDX[0].values)):
        myID = S_MYIDX[0].values[IDD]
        sctr = S_IDX.ix[S_IDX[0]==myID].sector.values[0]

        myID_sctr.append(sctr)

    return myID_sctr

####################################################################################
def retrn_df_sectors(myID_sctr,T2_index_SECTOR):
    """ make the T2_index_SECTOR DF columns be corresponding sector of the asset """

    T2_index_SECTOR.columns = myID_sctr

    return T2_index_SECTOR

####################################################################################
def return_df_of_sectors(S_DF, ID_SEC, dataF=True):

    """
    ID_SEC: Id of the sector I want
    S_DF: Sector dataframe; t2 index and columns => sector of asset
    -> Here we return the DF of a specific sector
    """

    SEC_X = []

    for i in range(len(S_DF)):
        sec = S_DF.ix[i][S_DF.columns == ID_SEC].values
        SEC_X.append(sec)

    # return daframe instead of pd.Series
    if dataF is True:
        SEC_X = pd.DataFrame(SEC_X)
        SEC_X.index = S_DF.index
    else:
        pass

    s,z = np.shape(SEC_X)
    print('sector [%s] has [%s] stocks'%(ID_SEC,z))

    return SEC_X

####################################################################################
def full_sec_DF(S_DF, mean=True, summ=False):

    # get unique sectors
    SECS = S_DF.columns.unique()

    SECTOR_DF_RIGHT = []

    for S in SECS:
        SEC_X = return_df_of_sectors(S_DF, S, dataF=True)
        if mean == True and summ == False:
            SEC_X = SEC_X.mean(axis=1)
        elif summ == True and mean == False:
            SEC_X = SEC_X.sum(axis=1)
        else:
            SEC_X = SEC_X.mean(axis=1)

        SECTOR_DF_RIGHT.append(SEC_X.values)

    # create the bigDataFrame
    SDF = pd.DataFrame(SECTOR_DF_RIGHT)
    SDF = SDF.T; SDF.index = S_DF.index; SDF.columns = SECS

    return SDF

####################################################################################
def get_sector_dataframe(T2_index, IDX, mean=True, summ=False):
    
    # Sort T2_INDEX
    T2_index_SECTOR = T2_index.reindex_axis(sorted(T2_index.columns), axis=1).copy()

    # My assets index
    S_MYIDX = pd.DataFrame(T2_index_SECTOR.columns)
    
    # sector Index
    S_IDX = create_key_sector_dataframe(IDX)
    
    # |F(1)|
    myID_sctr = retrn_sec_for_IDX_all(S_IDX,S_MYIDX)
    
    # |F(2)|
    S_DF = retrn_df_sectors(myID_sctr,T2_index_SECTOR)
    
    # create the [Nx11] Dataframe of compiled sectors
    SDF = full_sec_DF(S_DF, mean = mean, summ = summ)
    
    return SDF, S_DF, myID_sctr, S_MYIDX, S_IDX

####################################################################################
def price_DF_of_sector(INDX, S_IDX, data):

    """
    :param INDX: integer => containing the index number
    :param data: huge data frame with assets
    :return: DF of all assets from a given sector
    """

    p_tSeries = []

    # return a dataframe with assets that belong to that index INDX
    stcks_in_index = S_IDX[S_IDX.sector == INDX]

    # list the stocks of that sector
    stock_list = stcks_in_index[0].values

    for I in range(len(stock_list)):
        # Get the price time series of that asset
        p_tSeries_single_stock = data.ix[data.index==stock_list[I]]
        p_tSeries_single_stock.index = p_tSeries_single_stock.time
        p_tSeries_single_stock = p_tSeries_single_stock.drop(['time','adjclose','volume'], axis=1)
        # append results
        p_tSeries.append(p_tSeries_single_stock)

    # concatenate tudo
    DF = pd.concat(p_tSeries[0:-1],axis=1)

    # clean
    DF[DF==np.NaN] = 0
    DF = DF.sum(axis=1)

    return stcks_in_index, p_tSeries, DF

####################################################################################
def plot_sec_plus_ind(SDF, S_IDX, data, T2_index, init, last, savefig=True):

    # Get sectors
    cols = SDF.columns
    INDX = S_IDX.sector.unique()

    M_DF = []
    M_p_stck = []
    M_INDX_DF = []

    # Run function
    for sector in range(len(INDX)-2):
        INDX_DF, p_stck_df, DF = dfn.price_DF_of_sector(INDX[sector],
                                                        S_IDX, data)

        M_INDX_DF.append(INDX_DF)
        M_p_stck.append(p_stck_df)
        M_DF.append(DF)

    #plot
    f,ax = plt.subplots(3,3,figsize=(22,12))
    axs = ax.ravel()
    [dfn.plt_bsi_sector(T2_index, SDF[INDX[k]].values,
                        M_DF[k], init, last, cols[cols == INDX[k]].values[0],
                        hp_filter = True,
                        retAX=axs[k]) for k in range(len(cols)-2)]
    if savefig == True:
        plt.savefig('/Users/demos/Desktop/SBSI.pdf')

###################################################################################
###################################################################################

def data_munge_for_big_data_frame_qunzhi_data(results):

    import datetime as dt

    results['tc'] = (results['_tc'] - results['_t2']).dt.days
    results['dt'] = (results._t2 - results._t1).dt.days
    results.index = results._t2

    return results

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
                try:
                    cdf = cdf.ix[cdf['m']<1.]
                    cdf = cdf.ix[cdf['m']>0.1]
                    cdf = cdf.ix[cdf['B']<0.]
                    cdf = cdf.ix[cdf['tc']>=0]
                except:
                    cdf = cdf.ix[cdf['_m']<1.]
                    cdf = cdf.ix[cdf['_m']>0.1]
                    cdf = cdf.ix[cdf['_B']<0.]
                    cdf = cdf.ix[cdf['tc']>=0]
            # IF WE WANT NEGATIVE ONLY
            else:
                try:
                    cdf = cdf.ix[(cdf['_m']<1.)]
                    cdf = cdf.ix[cdf['_m']>0.1]
                    cdf = cdf.ix[cdf['_B']>=0.]
                    cdf = cdf.ix[cdf['tc']>=0]
                except:
                    cdf = cdf.ix[cdf['m']<1.]
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
                try:
                    cdf = cdf.ix[cdf['m']<0.9]
                    cdf = cdf.ix[cdf['m']>0.1]
                    cdf = cdf.ix[cdf['B']<0.]
                    cdf = cdf.ix[cdf['tc']>=t2]
                except:
                    cdf = cdf.ix[cdf['_m']<0.9]
                    cdf = cdf.ix[cdf['_m']>0.1]
                    cdf = cdf.ix[cdf['_B']<0.]
                    cdf = cdf.ix[cdf['_tc']>=t2]
            # IF WE WANT NEGATIVE ONLY
            else:
                try:
                    cdf = cdf.ix[(cdf['m']<0.9)]
                    cdf = cdf.ix[cdf['m']>0.1]
                    cdf = cdf.ix[cdf['B']>=0.]
                    cdf = cdf.ix[cdf['tc']>=t2]
                except:
                    cdf = cdf.ix[cdf['_m']<0.9]
                    cdf = cdf.ix[cdf['_m']>0.1]
                    cdf = cdf.ix[cdf['_B']<0.]
                    cdf = cdf.ix[cdf['_tc']>=t2]

            # size of dataframe post-filtering
            x_p = np.shape(cdf)

            # indicator
            res = x_p[0]/np.float(x[0])

            return count_number_of_timeScales_qualified(cdf)

    else:
        return np.array([0,0,0])

def count_number_of_timeScales_qualified(cdf):

    # Get dts
    dts_s = cdf.dt

    # total points per time_scale of interest
    #total_shrt  = 9.
    #total_med   = 10.
    #total_large = 14.
    total_shrt  = 57. 
    total_med   = 67.
    total_large = 115.

    # intervales
    shrt  = np.linspace(30, 200,(200-30)).astype(int)
    med   = np.linspace(200, 400,(400-200)).astype(int)
    large = np.linspace(400, 750,(750-400)).astype(int)

    #
    lrg_append, shrt_append, med_append = [],[],[]
    for i in range(len(cdf)):
        lrg_append.append(cdf.dt[i] in large)
        med_append.append(cdf.dt[i] in med)
        shrt_append.append(cdf.dt[i] in shrt)

    # compile
    total_largeScale_sigs = np.sum(lrg_append)/total_large
    total_medScale_sigs   = np.sum(med_append)/total_med
    total_shrtScale_sigs  = np.sum(shrt_append)/total_shrt



    return np.array([total_largeScale_sigs, total_medScale_sigs, total_shrtScale_sigs])

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

# plotting

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


def plot_3_indicators(pos_merged,G_SHORT,G_MED,G_LRG):

    f,a = plt.subplots(1,3,figsize=(16,4))
    pos_merged['close'].plot(color='k',ax=a[0])
    a[0].set_yticklabels('')
    b = a[0].twinx()
    G_LRG.plot(kind='area',color='g',ax=b,alpha=0.5)

    pos_merged['close'].plot(color='k',ax=a[1])
    a[1].set_yticklabels('')
    b = a[1].twinx()
    G_MED.plot(kind='area',color='y',ax=b,alpha=0.5)

    pos_merged['close'].plot(color='k',ax=a[2])
    a[2].set_yticklabels('')
    b = a[2].twinx()
    G_SHORT.plot(kind='area',color='r',ax=b,alpha=0.5)
    plt.tight_layout()


def plot_3_indicators_individualStock(pos_merged, G_SHORT, G_MED, G_LRG, tckr, y1, y2):

    lw = 1.6

    f,a = plt.subplots(3,1,figsize=(11,10))
    a[0].set_title(r'Asset: %s Large-scale bubble indicator'%tckr, fontsize=18)
    pos_merged[tckr][y1:y2].plot(color='k',ax=a[0], linewidth=lw)
    #a[0].set_yticklabels('')
    a[0].grid(True)
    b = a[0].twinx()
    b.grid()
    G_LRG[y1:y2].plot(kind='area',color='r',ax=b,alpha=0.5)
    G_LRG[y1:y2].plot(color='k',ax=b, linewidth=0.5)
    b.legend(['LPPLS-Indicator'], loc='upper left')

    a[1].set_title(r'Asset: %s Medium-scale bubble indicator'%tckr, fontsize=18)
    a[1].grid()
    pos_merged[tckr][y1:y2].plot(color='k',ax=a[1], linewidth=lw)
    b = a[1].twinx()
    G_MED[y1:y2].plot(kind='area',color='y',ax=b,alpha=0.5)
    G_MED[y1:y2].plot(color='k',ax=b, linewidth=0.5)
    b.legend(['LPPLS-Indicator'], loc='upper left')

    a[2].set_title(r'Asset: %s Short-scale bubble indicator'%tckr, fontsize=18)
    a[2].grid()
    pos_merged[tckr][y1:y2].plot(color='k',ax=a[2], linewidth=lw)
    #a[2].set_yticklabels('')
    b = a[2].twinx()
    G_SHORT[y1:y2].plot(kind='area',color='g',ax=b,alpha=0.5)
    G_SHORT[y1:y2].plot(color='k',ax=b, linewidth=0.5)
    b.legend(['LPPLS-Indicator'], loc='upper left')

    a[0].set_ylabel(r'$ln[P(t)]$', fontsize=18)
    a[1].set_ylabel(r'$ln[P(t)]$', fontsize=18)
    a[2].set_ylabel(r'$ln[P(t)]$', fontsize=18)

    a[1].grid(True)
    a[2].grid(True)

    plt.tight_layout()


def create_smoothed_indicators(pos_merged,neg_merged,negative=False):

    if negative == False:
        # Create the smoothed ind.
        G_SHORT = G_trust(0,pos_merged,shrt=True,med=False,lrg=False,plot=False)
        G_MED   = G_trust(0,pos_merged,shrt=False,med=True,lrg=False,plot=False)
        G_LRG   = G_trust(0,pos_merged,shrt=False,med=False,lrg=True,plot=False)
    else:
        G_SHORT = G_trust(0,neg_merged,shrt=True,med=False,lrg=False,plot=False)
        G_MED   = G_trust(0,neg_merged,shrt=False,med=True,lrg=False,plot=False)
        G_LRG   = G_trust(0,neg_merged,shrt=False,med=False,lrg=True,plot=False)

    return G_SHORT, G_MED, G_LRG


def get_hierarchical_indicators(DF, results):

    # Get all t2s
    t2s = results.index.unique()

    # Iterate over all t2's
    X_pos = [status_indicator_at_t2_hiearchy(t2, results) for t2 in t2s]
    X_neg = [status_indicator_at_t2_hiearchy(t2, results, pos = False) for t2 in t2s]

    # Construct positive and negative bbble indicators
    _trust = pd.DataFrame(X_pos,index=t2s,columns=['lrg','med','shrt'])
    _trust_neg = pd.DataFrame(X_neg,index=t2s,columns=['lrg','med','shrt'])

    # Merge with data
    pos_merged = pd.concat([DF,_trust],axis=1);
    #pos_merged = pos_merged.fillna(0)
    neg_merged = pd.concat([DF,_trust_neg],axis=1);
    #neg_merged = neg_merged.fillna(0)

    return pos_merged, neg_merged


def get_regular_trust_indicator_and_check_conditional_returns(results):
    # unique t2s
    T2S = results._t2.unique()

    # Iterate over all t2's
    X_pos = [fsl.status_indicator_at_t2(t2, results, pos=True, fromQz=True) for t2 in T2S]
    X_neg = [fsl.status_indicator_at_t2(t2, results, pos=False, fromQz=True) for t2 in T2S]

    # Construct positive and negative bbble indicators
    _trust = pd.DataFrame(X_pos,index=T2S,columns=['pos'])
    _trust_neg = pd.DataFrame(X_neg,index=T2S,columns=['neg'])

    return _trust, _trust_neg
