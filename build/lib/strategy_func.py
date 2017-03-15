#################################################################################
# Library for data handling for DS-DS project
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
import bt as bt
import ffn
from tabulate import tabulate

try:
    from numba import jit
except:
    def jit(func):
        return func

from pylppl_model import lppl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pylppl as lp
import sys
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/clean_codes/')
import data_functions as dfn

#######################################################################################################################
## Define paths
path_HD = '/Volumes/Demos BackupHD/Signals/0/'
path_bcktst_files = '/Users/demos/Desktop/BACKTESTING/'
path2 = '/Users/demos/bigdata/data_base/'
path2save = '/Volumes/Demos BackupHD/Signals/processed_data/'
path2save_shrt = '/Volumes/Demos BackupHD/Signals/processed_data_shrt/'
path2save_lng = '/Users/demos/Desktop/BACKTESTING/processed_data_lng/'

""" Functions to run the DS-DS trading strategy based on the LPPLS Bubble Indicator (c)
    - Functions I need
        - get bubble indicator value of each sector at time y
        - get assets from sector x
        - get top N performing assets of sector x
"""
#######################################################################################################################
def load_stbsi(offline=False):
    
    """ Load the SHRT_BSI while online or offline """
    
    if offline is True:
        try:
            ## location with individuals bubble status (assets)
            loc = '/Users/demos/Desktop/temp_files/'

            ## list directory
            FILES = os.listdir(loc)
        except:
            pass
            
        path_to_processed = '/Users/demos/Desktop/BACKTESTING/processed_data_shrt/'
        
        ## get benchmark
        Y = get_benchmark()

        ## get IDX
        IDX = dfn.get_idx_from_benchmark('/Users/demos/bigdata/data_base/')
        
        ## all stocks time series
        fname = '/Users/demos/Desktop/BACKTESTING/'+'DS_alldata_US.h5'
        data  = pd.read_hdf(fname,'data')


        ## load short-term MBSI: Market Bubble Status Indicator
        SHRT_MBSI = pd.read_hdf(path_to_processed+'SHRT_MBSI.h5','SHRT_MBSI')

        ## load short-term SBSI: Sector Bubble Status Indicator
        SHRT_SBSI = pd.read_hdf(path_to_processed+'SHRT_SDF.h5','SDF')

        ## load S_PrDF: Sector price time series
        # matrix (t2,n_setores) contendo o o price time series de each sec.
        SHRT_PrDF = pd.read_hdf(path_to_processed+'SHRT_S_PrDF.h5','SHRT_S_PrDF')

        ## load S_PrDF: Assets within each sector DataFrame
        # matrix (t2,n_setores) contendo o asset price time series de each sec.
        # Ex: for acessing -> X[SECTOR][0] with SECTOR \in [0:8]
        Asst_perSector_DF = pd.read_hdf(path_to_processed+'Asst_perSector_DF.h5',
                                        'Asst_perSector_DF')
        
    elif offline is not True:
        try:
            ## location with individuals bubble status (assets)
            loc = '/Users/demos/Desktop/temp_files/'

            ## list directory
            FILES = os.listdir(loc)
        except:
            pass
        
        ## get benchmark
        Y = get_benchmark()

        ## get IDX
        IDX = dfn.get_idx_from_benchmark('/Users/demos/bigdata/data_base/')
        
        ## all stocks time series
        fname = '/Users/demos/Desktop/BACKTESTING/'+'DS_alldata_US.h5'
        data  = pd.read_hdf(fname,'data')


        ## load short-term MBSI: Market Bubble Status Indicator
        SHRT_MBSI = pd.read_hdf(path2save_shrt+'SHRT_MBSI.h5','SHRT_MBSI')

        ## load short-term SBSI: Sector Bubble Status Indicator
        SHRT_SBSI = pd.read_hdf(path2save_shrt+'SHRT_SDF.h5','SDF')

        ## load S_PrDF: Sector price time series
        # matrix (t2,n_setores) contendo o o price time series de each sec.
        SHRT_PrDF = pd.read_hdf(path2save_shrt+'SHRT_S_PrDF.h5','SHRT_S_PrDF')

        ## load S_PrDF: Assets within each sector DataFrame
        # matrix (t2,n_setores) contendo o asset price time series de each sec.
        # Ex: for acessing -> X[SECTOR][0] with SECTOR \in [0:8]
        Asst_perSector_DF = pd.read_hdf(path2save_shrt+'Asst_perSector_DF.h5',
                                        'Asst_perSector_DF')
        
        
    return Y, IDX, data, SHRT_SBSI, SHRT_MBSI, SHRT_PrDF, Asst_perSector_DF

#######################################################################################################################
def load_ltbsi():

    ## location with individuals bubble status
    #loc = '/Users/demos/Desktop/temp_files/'

    ## list directory
    #FILES = os.listdir(loc)

    ## get benchmark
    Y = get_benchmark()

    ## get IDX
    IDX = dfn.get_idx_from_benchmark(path2)

    ## load MBSI: Market Bubble Status Indicator
    # matrix (t2,1) contendo o bubble status indicator para Market.
    MBSI = pd.read_hdf(path2save_lng+'MBSI.h5','MBSI')

    ## load SBSI: Sector Bubble Status Indicator
    # matrix (t2,n_setores) contendo o bubble status indicator para each sec.
    SBSI = pd.read_hdf(path2save_lng+'SBSI.h5','SBSI')

    ## load S_PrDF: Sector price time series
    # matrix (t2,n_setores) contendo o o price time series de each sec.
    S_PrDF = pd.read_hdf(path2save_lng+'S_PrDF.h5','TY')

    ## load S_PrDF: Assets within each sector DataFrame
    # matrix (t2,n_setores) contendo o asset price time series de each sec.
    # Ex: for acessing -> X[SECTOR][0] with SECTOR \in [0:8]
    Asst_perSector_DF = pd.read_hdf(path2save_lng+'Asst_perSector_DF.h5',
                                    'Asst_perSector_DF')

    ## all stocks time series
    fname = '/Users/demos/Desktop/BACKTESTING/'+'DS_alldata_US.h5'
    data  = pd.read_hdf(fname,'data')

    return Y, IDX, MBSI, SBSI, S_PrDF, Asst_perSector_DF, data

#######################################################################################################################
def get_benchmark():
    bnchmrk = pd.read_hdf(path_bcktst_files+'DS_benchmark.h5','sp500')
    Y = pd.DataFrame(bnchmrk)

    return Y
#######################################################################################################################
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
    DF = DF.mean(axis=1) ### MUDEI AQUI DE SUM PARA MEAN !!!

    return stcks_in_index, p_tSeries, DF
#######################################################################################################################
def get_timeSeries_from_sectors(SDF, S_IDX, data):

    ## load S_PrDF: Sector price time series
    #  matrix (t2,n_setores) contendo o o price time series de each sec.

    # Get sectors
    cols = SDF.columns
    INDX = S_IDX.sector.unique()

    M_DF = []
    M_p_stck = []
    M_INDX_DF = []

    # Run function
    for sector in range(len(INDX)-2):
        INDX_DF, p_stck_df, DF = price_DF_of_sector(INDX[sector],
                                                    S_IDX, data)

        M_INDX_DF.append(INDX_DF)
        M_p_stck.append(p_stck_df)
        M_DF.append(DF)

    return M_INDX_DF, M_p_stck, M_DF
#######################################################################################################################
def get_price_dataframe_from_all_sectors(S_IDX, M_p_stck):

    # matrix (t2,n_setores) contendo o asset price time series de each sec.
    # Ex: for acessing -> X[SECTOR][0] with SECTOR \in [0:8]

    # # #
    DFSS = []
    AsstsPerSecDF = []

    for SEC in range(len(M_p_stck)):

        # # #
        Cols_from_secX = S_IDX.ix[S_IDX.sector == S_IDX.sector.unique()[SEC]][0].values

        # # #
        DFSS = pd.concat(M_p_stck[SEC][0:-1],axis=1)
        DFSS.columns = Cols_from_secX[0:-1]

        AsstsPerSecDF.append(DFSS)

    return AsstsPerSecDF
#######################################################################################################################
def ret_GoodSIndx(T2, SDF):

    # return the Sectors with positive SBSI array(X)
    X = SDF.ix[T2][SDF.ix[T2] > 0.]
    X = np.array(X.axes[0][0:-1])

    return X
#######################################################################################################################
def classical_drawdowns(prices, return_sizes=True):
    """ Extract classical drawdowns (epsilon-drawdowns with epsilon=0) from the
        numpy.array with prices.

        Returns numpy array with turning points that indicate beginning of trends
        (+1 for upward trend and -1 for downward trend). Thus dynamics between
        +1 and -1 corresponds to draw-up and dynamics between -1 and +1
        corresponds to drawdown.

        If return_sizes is True, then instead of indicators, sizes of respective
        drawdowns and drawups will be returned. Then indicators could be obtained
        with the sign function of the output.

        NOTE: the last value should be better ignored (for turning points it
        only indicates end of time series and for sizes its value is arbitrary).
    """
    sizes = _calc_classical_drawdowns(np.array(prices))
    if return_sizes:
        return sizes
    else:
        return np.sign(sizes)
#######################################################################################################################
def _calc_classical_drawdowns(prices):
    ret = np.diff(prices)
    sizes = np.zeros(len(ret))

    sgn = 0  # current sign of drawdown/drawup
    runsum = 0.  # running sum
    last_ii = 0  # index of start of last drawdown/drawup
    for ii in range(len(ret)):
        s = np.sign(ret[ii])
        if (s != sgn) and (s != 0):
            sgn = s

            sizes[last_ii] = runsum
            last_ii = ii
            runsum = 0.
        runsum += ret[ii]

    sizes[-1] = runsum
    sizes = np.r_[sizes, -sgn]
    return sizes

########################################################################################################################
def run_strategy(sPr, sBsi, mBsi, rule_to_rebalance, fillna=True):

    """
    :param sPr: sector price dataframe
    :return:
    """

    ret = sPr.diff().copy()
    ret_good = (sBsi>0).copy()

    # drop two sectors
    ret_good = ret_good.drop([6000],axis=1)
    ret_good = ret_good.drop([116],axis=1)

    ## Run the strategy
    PORTF = []
    for T2 in sBsi.index:
        # if MBSI > 0:
        if mBsi.ix[T2].values[0] > rule_to_rebalance:
            X = ret_GoodSIndx(T2, sBsi) # retrieve good indexes at T2 (good if sBsi > 0)
            # get all the time series of these indexes up to MBSI < 0:
            N = np.float(np.size(X))
            PORTF.append( (1/N) * ret.ix[T2][X])
        elif mBsi.ix[T2].values[0] <= rule_to_rebalance:
            X = ret_GoodSIndx(T2, sBsi)
            PORTF.append(0. * ret.ix[T2][X])

    ## CALCULATIONS
    PRT = pd.DataFrame(PORTF)
    if fillna == True:
        PRT = PRT.fillna(0)
    else:
        pass
    try:
        PRT = PRT.mean(axis=1)
        PRT = pd.DataFrame(PRT)
        PRT.ix[str(PRT.index[0]):'1973-02-09'] = 0
    except:
        pass

    return PRT

#######################################################################################################################
def plot_res(res, data, bnchmrk, fname=None):

    """
    Trading strategy results:
    res: results from the trading strategy
    data: assets to perform the trading strategy

    """

    bst = res.backtests['s0']
    bst.stats.set_riskfree_rate(0.025/252)
    p = bst.stats.prices
    d = bst.stats.drawdown
    h = bst.herfindahl_index

    #((bst.weights>0).any()).sum()
    D = pd.DataFrame(d)

    f = plt.figure(1,figsize=(11,7)) #create one of the figures that must appear with the chart
    plt.grid(False)
    gs = gridspec.GridSpec(14,1)
    gs.update(hspace=0)
    ax = plt.subplot(gs[:11, :])
    p.plot(grid=False, ax=ax, legend=False)
    ax2 = ax.twinx()
    bnchmrk.plot(grid=False, color='red',legend=False,ax=ax2)
    ax2.set_yticklabels([''])

    ax3 = ax.twinx()
    D.plot(grid=False, kind="area",ax=ax3,alpha=0.15,color='blue',legend=False)
    ax3.set_ylabel(r'$Drawdowns$',fontsize=13)
    ax3.set_xlabel('')

    ax1 = plt.subplot(gs[12:, :])
    ax1.plot(h.index, (1/h).values,'b', lw=0.7)
    ax1.grid(True)
    ax1.set_ylabel('Effective num.\n of stocks')
    ax1.set_xlim([p.index[0],p.index[-1]])

    # DETAILS
    ax2.set_xlabel(''); ax2.set_xticklabels('')
    ax.set_title(header_stats(bst),family='monospace', multialignment='left', fontsize=9)
    ax.grid(True, which='major', axis='both')
    ax.axhline(100, color='k', ls='--', lw=0.5)
    ax.set_xticklabels(''); ax.set_xlabel('')
    ax.set_ylabel(r'$P(t)$',fontsize=13)

    # legend
    blue_line    = mlines.Line2D([], [], color='blue', linestyle='-', label=r'$Portfolio $')
    red_line     = mlines.Line2D([], [], color='red', linestyle='-', label=r'$Benchmark\, (S&P500)$')
    ax.legend(handles=[blue_line,red_line],loc='lower right',fontsize=13)

    plt.subplots_adjust(hspace=0.05, wspace=0.1)

    plt.tight_layout()

    if fname is not None:
        plt.savefig('/Users/demos/Desktop/BACKTESTING/2 show DS/res_strategies/'+fname+'.pdf')

#######################################################################################################################
def plot_dd_y_port(PRT, Y, init, last, title='STBSI'):

    ## Drawdowns
    st = PRT.calc_stats()
    CS = st.prices.cumsum()
    CS.ix['1990':'1991'] = 0
    DD = CS.to_drawdown_series()

    ##### figure
    fig, ax = plt.subplots(figsize=(10,5))

    # Twin the x-axis twice to make independent y-axes.
    axes = [ax, ax.twinx(), ax.twinx()]

    # Make some space on the right side for the extra y-axis.
    fig.subplots_adjust(right=0.75)

    #
    plt.title(title)

    # Move the last y-axis spine over to the right by 20% of the width of the axes
    axes[-1].spines['right'].set_position(('axes', 1.15))
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)

    ## plot
    PRT[init:last].cumsum().plot(color='b', legend=False, ax=axes[0])
    Y[init:last].plot(color='r', legend=False, ax=axes[1])
    DD[init:last].plot(kind='area', color='blue', alpha=0.4,
                       ax=axes[2], legend=False)
    axes[2].set_ylim([-.50,0])

    ## labels
    axes[0].set_ylabel(r'$Portfolio$',fontsize=20)
    axes[1].set_ylabel(r'$S&P500$',fontsize=20)
    axes[2].set_ylabel(r'$DD$',fontsize=20)

    ## axis
    #axes[0].set_ylim([0,2500])
    #axes[1].set_ylim([0,2500])

    plt.tight_layout()
#######################################################################################################################
def plot_sec_bsi(SHRT_PrDF, SHRT_SBSI, INDX, ax=None):

    if ax is not None:
        SHRT_PrDF[INDX].plot(color='black',ax=ax)
        ax.set_title('SECTOR: %s'%INDX)
        a = ax.twinx()
        SHRT_SBSI[INDX].plot(color='blue',linewidth=0.8,alpha=0.5,ax=a)
        a.axhline(0,color='black',linestyle='--',linewidth=1.5)
        plt.tight_layout()
    else:
        SHRT_PrDF[INDX].plot(color='black')
#######################################################################################################################
def plt_all_sectors_sbsi(SHRT_PrDF, SHRT_SBSI):

    """
    EX: sdfn = reload(sdfn)
    sdfn.plt_all_sectors_sbsi(SHRT_PrDF, SHRT_SBSI)
    """

    INDX = SHRT_PrDF.columns

    f,ax = plt.subplots(3,3,figsize=(22,12))
    axs = ax.ravel()
    [plot_sec_bsi(SHRT_PrDF, SHRT_SBSI, INDX[k], ax=axs[k]) for k in range(len(INDX))]

    plt.tight_layout()
#######################################################################################################################
def times_we_cross_zero_mbsi(mBsi):
    X = [np.sign(mBsi.ix[k]) == np.sign(mBsi.ix[k-1]) for k in range(len(mBsi.index))]
    
    C = []

    for i in range(len(X)):
        if X[i][0] == False:
            C.append(1)
        else:
            C.append(0)
            
    return np.sum(C)
#######################################################################################################################
def series_to_backtest(sBsi, sPr):
    
    """ Get time series correctly to perform the backtesting using BT """
    
    fuzzy_t2 = pd.DataFrame(sPr.index,index=[sPr.index],columns=['t2'])
    RIGHT = pd.concat([fuzzy_t2,sBsi],axis=1)
    RIGHT = RIGHT.drop(['t2'],axis=1)
    
    signal = RIGHT>0
    
    return signal
#######################################################################################################################
def clean_sector_tseries_2_backtest(sBsi, sPr):

    try:
        sBsi = sBsi.drop([6000],axis=1)
        sBsi = sBsi.drop([116],axis=1)
        sPr = sPr.drop([6000],axis=1)
        sPr = sPr.drop([116],axis=1)
    except:
        pass

    signal = series_to_backtest(sBsi, sPr)

    return signal
#######################################################################################################################
def run_stbsi_strategy(sPr, signal, comissions=True):

    fees = lambda q, p: max(1., abs(q) * p * 0.25/100.)

    ss = bt.Strategy('s0',[bt.algos.RunDaily(),
                            bt.algos.SelectWhere(signal==True),
                            #bt.algos.SelectThese(signal),
                            bt.algos.WeighEqually(),
                            bt.algos.Rebalance()])

    if comissions == True:
        test = bt.Backtest(ss, sPr, commissions=fees)
        res = bt.run(test)
    else:
        test = bt.Backtest(ss, sPr)
        res = bt.run(test)

    return res
#######################################################################################################################
def get_signal_mbsi_and_sbsi_test2(cond1, cond2, sPr, sBsi, mBsi, conditions_diff=False):

    """
    :param cond1: threshold for the sector bubble status indicator
    :param cond2: threshold for the market bubble status indicator
    :param sPr: sector price time series
    :param sBsi: sector bubble status indicator
    :return:
    """

    try:
        sBsi = sBsi.drop([6000],axis=1)
        sBsi = sBsi.drop([116],axis=1)
        sPr = sPr.drop([6000],axis=1)
        sPr = sPr.drop([116],axis=1)
    except:
        pass

    #### ENTENDER ESSE METODO DE CONCATENAMENTO !O!O!O!O!O!
    # Firs we chek (boolean) the sectors whose sBsi >= cond1 (=0.)
    fuzzy_t2 = pd.DataFrame(sPr.index,index=[sPr.index],columns=['t2'])

    # Old way for concatenating
    RIGHT = pd.concat([fuzzy_t2,sBsi],axis=1)
    RIGHT = RIGHT.drop(['t2'],axis=1).copy()
    # RIGHT IS A BOOLEAN VECTOR WHERE THE SECTORS ARE > 0.

    fuzzy_t2 = pd.DataFrame(sPr.index,index=[sPr.index],columns=['t2'])
    LEFT = pd.concat([fuzzy_t2,mBsi],axis=1)
    LEFT = LEFT.drop(['t2'],axis=1).copy()

    if conditions_diff == True:
        RIGHT_BAD = (RIGHT >= -0.2) & (RIGHT <= 0.6)
        LEFT_BAD = (LEFT >= -0.2) & (LEFT <= 0.6)
    else:
        LEFT_BAD = (LEFT > cond2).copy()
        RIGHT_BAD = (RIGHT > cond1).copy()

    u,p = np.shape(RIGHT)

    for I in range(u):
        if LEFT_BAD.ix[I].any() == False:
            RIGHT_BAD.ix[I] = False
        else:
            continue

    signal = RIGHT_BAD.copy()

    return signal, RIGHT, LEFT_BAD
#######################################################################################################################
# NEW STRATEGY MAXIMIZATION Nov. 3
#######################################################################################################################
def get_sorted_assets_at_given_year_from_given_sector(df, T2):

    # get the range of years
    _years = [d.strftime('%Y') for d in pd.date_range('1973','2015',freq='12M')]

    # how many years we have
    #print('number of years within data set = %s'%np.shape(_years))

    # my dataFrame at year[X]
    df[str(_years[T2])].head()

    # mean performance of each asset
    T = pd.DataFrame(df[str(_years[T2])].mean(), columns=[_years[T2]]);

    # sort the values from top to bottom
    T = T.sort_values(_years[T2], ascending=False, na_position='last')

    return T, _years

#######################################################################################################################
def get_sorted_assets_var(df, T2):

    # get the range of years
    _years = [d.strftime('%Y') for d in pd.date_range('1973','2015',freq='12M')]

    # how many years we have
    #print('number of years within data set = %s'%np.shape(_years))

    # my dataFrame at year[X]
    df[str(_years[T2])].head()

    # mean performance of each asset
    T = pd.DataFrame(df[str(_years[T2])].var(), columns=[_years[T2]]);

    # sort the values from top to bottom
    T = T.sort_values(_years[T2], ascending=False, na_position='last')

    return T, _years

#################################################################################
def get_best_performers_at_year(df,year,mean=True):


    """
    Pick the top performing assets of the previous year. THat is, we rank assets according to mean(ret)
    choose year('xxxx-xx-xx') and the df => contain assets that belong to a given sector choosed previously
    If one wants all results for a given sector, just pick TT
    """

    # T2 /in [0,41]
    T2 = np.arange(42)
    TT2 = pd.DataFrame(T2)

    # append all results per year
    TT = []

    if mean is not False:
        for Year in T2:
            T,_years = get_sorted_assets_at_given_year_from_given_sector(df, Year)
            TT.append(T)
    else:
        for Year in T2:
            T,_years = get_sorted_assets_var(df, Year)
            TT.append(T)

    TT2.index = _years
    _num_loc_of_index  = TT2.ix[year][0]
    test = pd.DataFrame(TT)
    DF   = pd.DataFrame(test[0][_num_loc_of_index])

    return DF, TT

#################################################################################
def run_it_all_2(sPr,sBsi,mBsi):

    # condition for the Sector bubble status and cond2 for the mbsi
    cond1, cond2 = np.linspace(-0.5,0.5,10,dtype=float), np.linspace(-0.5,0.5,5,dtype=float)

    DD   = []; DD2 = []; CAGR = []; CAGR2 = []; H    = []; H2   = []; SHRP = []; SHRP2 = []

    for conD2 in cond2:
        for conD in cond1:
            # Get the signals for running strategy
            signal, RIGHT, LEFT_BAD = get_signal_mbsi_and_sbsi_test2(conD,
                                                                     conD2,
                                                                     sPr,
                                                                     sBsi,
                                                                     mBsi,
                                                                     conditions_diff=False)
            # run the strategy
            res = run_stbsi_strategy(sPr['1990':'2015'],
                                     signal['1990':'2015'],
                                     comissions=True)

            # calc statistics
            bst = res.backtests['s0']
            bst.stats.set_riskfree_rate(0.025)

            dd    = bst.stats.max_drawdown
            cagr  = bst.stats.cagr
            h     = np.mean(bst.herfindahl_index)
            shrpY = res.stats.ix['yearly_sharpe'][0]

            DD.append(dd)
            CAGR.append(cagr)
            H.append(h)
            SHRP.append(shrpY)

        DD2.append(DD)
        CAGR2.append(CAGR)
        H2.append(H)
        SHRP2.append(SHRP)

    return DD2,CAGR2,H2,SHRP2, cond1, cond2

#################################################################################
def plot_performance_as_function_of_BSI(cond1,DD,H,CAGR,SHRP,savefig=True):


    """Shitty way: Change !"""

    f,((ax,ax1),(ax2,ax3)) = plt.subplots(2,2,figsize=(13,6),
                                          sharex=True)

    ax.plot(cond1,DD[0][0:10],'ks--'); ax.grid(True)
    ax.plot(cond1,DD[0][10:20],'bs--'); ax.grid(True)
    ax.plot(cond1,DD[0][20:30],'gs--'); ax.grid(True)
    ax.plot(cond1,DD[0][30:40],'cs--'); ax.grid(True)
    ax.plot(cond1,DD[0][40:50],'ms--'); ax.grid(True)
    #ax.set_xlabel('S_BSI threshold')
    ax.set_ylabel('Max. DD size')

    ax1.plot(cond1,H[0][0:10],'ks--'); ax1.grid(True)
    ax1.plot(cond1,H[0][10:20],'bs--'); ax.grid(True)
    ax1.plot(cond1,H[0][20:30],'gs--'); ax.grid(True)
    ax1.plot(cond1,H[0][30:40],'cs--'); ax.grid(True)
    ax1.plot(cond1,H[0][40:50],'ms--'); ax.grid(True)
    #ax1.set_xlabel('S_BSI threshold')
    ax1.set_ylabel('$\mu(H)$')

    ax2.plot(cond1,CAGR[0][0:10],'ks--'); ax2.grid(True)
    ax2.plot(cond1,CAGR[0][10:20],'bs--'); ax.grid(True)
    ax2.plot(cond1,CAGR[0][20:30],'gs--'); ax.grid(True)
    ax2.plot(cond1,CAGR[0][30:40],'cs--'); ax.grid(True)
    ax2.plot(cond1,CAGR[0][40:50],'ms--'); ax.grid(True)
    ax2.set_xlabel('threshold_1')
    ax2.set_ylabel('CAGR')

    ax3.plot(cond1,SHRP[0][0:10],'ks--'); ax3.grid(True)
    ax3.plot(cond1,SHRP[0][10:20],'bs--'); ax.grid(True)
    ax3.plot(cond1,SHRP[0][20:30],'gs--'); ax.grid(True)
    ax3.plot(cond1,SHRP[0][30:40],'cs--'); ax.grid(True)
    ax3.plot(cond1,SHRP[0][40:50],'ms--'); ax.grid(True)
    ax3.set_xlabel('threshold_1')
    ax3.set_ylabel('Yearly Sharpe Ratio')

    ## legend ax3
    one   = mlines.Line2D([], [], color='black', marker='s',
                          linestyle='-', label=r'$thrshld_2 = -0.50$')
    two   = mlines.Line2D([], [], color='blue', marker='s',
                          linestyle='-', label=r'$thrshld_2 = -0.25$')
    three = mlines.Line2D([], [], color='green', marker='s',
                          linestyle='-', label=r'$thrshld_2 = 0.00$')
    four  = mlines.Line2D([], [], color='cyan', marker='s',
                          linestyle='-', label=r'$thrshld_2 = 0.25$')
    five  = mlines.Line2D([], [], color='magenta', marker='s',
                          linestyle='-', label=r'$thrshld_2 = 0.50$')
    ax2.legend(handles=[one, two, three, four, five],
               loc='lower left',fontsize=12)

    plt.tight_layout()

    if savefig == True:
        plt.savefig('/Users/demos/Desktop/test_yes.pdf')
    else:
        pass

#################################################################################
def signal_strategy_3(SHRT_MBSI, SHRT_SBSI, LT_SBSI, LT_MBSI,
                      S_PrDF, thr_ewma, thr_sctr, ma_days,
                      strategy1 = True):

    """ Get signals using strategy 1
        from Qunzhi's new one

        strategy 1:
        strategy 2:

    """

    sPr  = S_PrDF.copy()
    mBsi = SHRT_MBSI.copy() # short term market bubble status ind.
    sBsi = SHRT_SBSI.copy() # short-term sector bsi
    ltSbsi = LT_SBSI.copy() # long-term sector bsi
    ltMbsi = LT_MBSI.copy() # long-term market bsi

    mBsi = modify_size_matrices_2_match(mBsi,sPr)
    sBsi = modify_size_matrices_2_match(sBsi,sPr)
    ltSbsi = modify_size_matrices_2_match(ltSbsi,sPr)
    ltMbsi = modify_size_matrices_2_match(ltMbsi,sPr)

    try:
        sBsi = sBsi.drop([6000],axis=1)
        sBsi = sBsi.drop([116],axis=1)
        sPr  = sPr.drop([6000],axis=1)
        sPr  = sPr.drop([116],axis=1)
        mBsi = mBsi.drop([6000],axis=1)
        mBsi = mBsi.drop([116],axis=1)
        ltSbsi  = ltSbsi.drop([6000],axis=1)
        ltSbsi  = ltSbsi.drop([116],axis=1)
        ltMbsi = ltMbsi.drop([6000],axis=1)
        ltMbsi = ltMbsi.drop([116],axis=1)
        signal3 = signal3.drop([6000],axis=1)
        signal3 = signal3.drop([116],axis=1)
    except:
        pass

    # indicator for leaving the market
    ind1 = mBsi < - 0.05

    # indicator for entering the market
    ind_enter = mBsi > 0.

    if strategy1 is not False:
        # consideration regarding sectors
        EWMA_SEC = pd.rolling_mean(ltSbsi,ma_days)
        test = (ltSbsi-EWMA_SEC)
    else:
        # Run strategy two which is the same as 1 but withou long-term bubble indicator
        # consideration regarding sectors
        EWMA_SEC = pd.rolling_mean(sBsi,ma_days)
        test = (sBsi-EWMA_SEC)

    # condition1 -> (sector_status-EWMA) > 0.05
    cond1 = test >= thr_ewma
    # condition2 -> sector_status > 0.05
    cond2 = SHRT_SBSI > thr_sctr

    # signal
    signal3 = (cond1 & cond2)

    # clean signal
    signal3 = signal3.drop([6000],axis=1)
    signal3 = signal3.drop([116],axis=1)

    signal3 = pd.concat([signal3,sPr[4000]],axis=1)
    signal3 = signal3.drop([4000],axis=1)
    signal3 = signal3.fillna(False).copy()

    return signal3, cond1, cond2, ind_enter

#################################################################################
def modify_size_matrices_2_match(price,sPr):

    """ enter a matrix [sPr] and get it with a new index  """

    # Firs we chek (boolean) the sectors whose sBsi >= cond1 (=0.)
    fuzzy_t2 = pd.DataFrame(sPr.index,index=[sPr.index],columns=['t2'])
    RIGHT = pd.concat([fuzzy_t2,price],axis=1)
    RIGHT = RIGHT.drop(['t2'],axis=1).copy()

    return RIGHT

#################################################################################
def run_it_all_3(sPr, SHRT_MBSI,SHRT_SBSI, LT_SBSI, LT_MBSI, S_PrDF, cond1, cond2, ma_days):

    DD   = []; DD2 = []; CAGR = []; CAGR2 = []; H    = []; H2   = []; SHRP = []; SHRP2 = []

    for thr_sctr in cond2:
        for thr_ewma in cond1:
            ### get signal
            sig, _, _, ind_enter = signal_strategy_3(SHRT_MBSI,
                                                     SHRT_SBSI, LT_SBSI, LT_MBSI,
                                                     S_PrDF,thr_ewma,thr_sctr,ma_days)

            ### step 2
            for i in range(len(sig.index)):
                if ind_enter.ix[i].any() == False:
                    sig.ix[i] = False
                else:
                    pass

            ### run the strategy
            sPr  = S_PrDF.copy()
            res = run_stbsi_strategy(sPr['1990':'2015'],
                                     sig['1990':'2015'],
                                     comissions=False)

            # calc statistics
            bst = res.backtests['s0']
            bst.stats.set_riskfree_rate(0.025)

            dd    = bst.stats.max_drawdown
            cagr  = bst.stats.cagr
            h     = np.mean(bst.herfindahl_index)
            shrpY = res.stats.ix['yearly_sharpe'][0]

            DD.append(dd)
            CAGR.append(cagr)
            H.append(h)
            SHRP.append(shrpY)

        DD2.append(DD)
        CAGR2.append(CAGR)
        H2.append(H)
        SHRP2.append(SHRP)

    return DD2,CAGR2,H2,SHRP2

#################################################################################
def run_trategy_3(SHRT_MBSI, SHRT_SBSI,LT_SBSI, LT_MBSI, S_PrDF, thr_ewma, thr_sctr, ma_days, com=True):

    """ Here we get the signal for strategy 3 and run the strategy """


    ### get signal
    sig, cond1, cond2, ind_enter = signal_strategy_3(SHRT_MBSI, SHRT_SBSI,LT_SBSI,
                                                     LT_MBSI, S_PrDF,thr_ewma, thr_sctr, ma_days)

    ### step 2
    for i in range(len(sig.index)):
        if ind_enter.ix[i].any() == False:
            sig.ix[i] = False
        else:
            pass

    ### run the strategy
    sPr  = S_PrDF.copy()
    if com == True:
        res = run_stbsi_strategy(sPr['1990':'2015'],sig['1990':'2015'],comissions=True)
    else:
        res = run_stbsi_strategy(sPr['1990':'2015'],sig['1990':'2015'],comissions=False)

    return sig, res


#################################################################################
def header_stats(backtest, bench=None):
    """
    Prepare header for figure with stats of backtesting.
    """
    stats_t1 = [
         ('total_return', 'Total Return', 'p'),
        ('daily_sharpe', 'Daily Sharpe', 'n'),
        ('cagr', 'CAGR', 'p'),
        ('max_drawdown', 'Max Drawdown', 'p'),
        ('avg_drawdown', 'Avg. DrDwn', 'p'),
        ('avg_drawdown_days', 'Avg. DrDwn Days', 'n')]
    stats_t2 = [
        ('monthly_sharpe', 'Monthly Sharpe', 'n'),
        ('monthly_mean', 'Monthly Mean (ann)', 'p'),
        ('monthly_vol', 'Monthly Vol (ann)', 'p'),
        ('best_month', 'Best Month', 'p'),
        ('worst_month', 'Worst Month', 'p')]
    stats_t3 = [
        ('yearly_sharpe', 'Yearly Sharpe', 'n'),
        ('yearly_mean', 'Yearly Mean', 'p'),
        ('yearly_vol', 'Yearly Vol', 'p'),
        ('best_year', 'Best Year', 'p'),
        ('worst_year', 'Worst Year', 'p')]

    align_str = lambda st: [x.ljust(max([len(x) for x in st])) for x in st]

    st1 = align_str(table_backtest_stats(backtest, stats_t1).split('\n'))
    st2 = align_str(table_backtest_stats(backtest, stats_t2).split('\n'))
    st3 = align_str(table_backtest_stats(backtest, stats_t3).split('\n'))

    if bench is not None:
        stats_t1b = [(a, '', c) for (a, b, c) in stats_t1]
        stb = align_str(table_backtest_stats(bench, stats_t1b).split('\n'))

    st_all = []
    if bench is None:
        for (s1, s2, s3) in zip(st1, st2, st3):
            st_all.append(s1 + '    ' + s2 + '    ' + s3)
    else:
        for (s1, s2, s3, sb) in zip(st1, st2, st3, stb):
            st_all.append(s1 + ' (BM: ' + sb.lstrip() + ' )   ' + s2 + '    ' + s3)

    return '\n'.join(align_str(st_all))

def table_backtest_stats(backtest, stat_template, header=False):
    """
    Prepare backtest stats for given template
    """
    stats = backtest.stats

    data = []
    first_row = ['Stat']
    first_row.extend(['Value'])
    data.append(first_row)

    for stat in stat_template:
        k, n, f = stat
        # blank row
        if k is None:
            row = [''] * len(data[0])
            data.append(row)
            continue

        row = [n]
        raw = getattr(stats, k)
        if f is None:
            row.append(raw)
        elif f == 'p':
            row.append(ffn.fmtp(raw))
        elif f == 'n':
            row.append(ffn.fmtn(raw))
        elif f == 'dt':
            row.append(raw.strftime('%Y-%m-%d'))
        else:
            raise NotImplementedError('unsupported format %s' % f)
        data.append(row)

    res = tabulate(data, headers='firstrow')
    if not header:
        res = res[res.find('-\n')+2:]
    return res
#################################################################################
# TESTING THE MOVING AVERAGE

def run_diff_EWMA(sPr, SHRT_MBSI, SHRT_SBSI, LT_SBSI, LT_MBSI, S_PrDF, thr_ewma,thr_sctr, ma_days):

    DD   = []; DD2 = []; CAGR = []; CAGR2 = []; H    = []; H2   = []; SHRP = []; SHRP2 = []

    ### get signal
    for MA in ma_days:
        sig, _, _, ind_enter = signal_strategy_3(SHRT_MBSI,
                                                      SHRT_SBSI, LT_SBSI, LT_MBSI,
                                                      S_PrDF,thr_ewma,thr_sctr,MA)

        ### step 2
        for i in range(len(sig.index)):
            if ind_enter.ix[i].any() == False:
                sig.ix[i] = False
            else:
                pass

        ### run the strategy
        sPr  = S_PrDF.copy()
        res = run_stbsi_strategy(sPr['1990':'2015'],
                                 sig['1990':'2015'],
                                 comissions=False)

        # calc statistics
        bst = res.backtests['s0']
        bst.stats.set_riskfree_rate(0.025)

        dd    = bst.stats.max_drawdown
        cagr  = bst.stats.cagr
        h     = np.mean(bst.herfindahl_index)
        shrpY = res.stats.ix['yearly_sharpe'][0]

        DD.append(dd)
        CAGR.append(cagr)
        H.append(h)
        SHRP.append(shrpY)

    return DD,CAGR,H,SHRP

#################################################################################
def plot_ewma_res(DD,CAGR,H,SHRP,ma_days):
    RES = pd.concat([pd.DataFrame(DD,columns=['DD']),pd.DataFrame(CAGR,columns=['CAGR']),
                 pd.DataFrame(H,columns=['H']),
                 pd.DataFrame(SHRP,columns=['SHRP'])],
                axis=1)

    RES.index=ma_days

    f,ax = plt.subplots(4,1,figsize=(7,7))
    RES.plot(subplots=True,marker='s',ax=ax)
    plt.tight_layout()
#################################################################################
def plot_4_(STsPr,CONC):

    """CONC = condition boolean """


    f,((ax0,ax1),(ax2,ax3),(ax4,ax5),(ax6,ax7)) = plt.subplots(4,2,figsize=(9,10))

    CONC[4000].plot(kind="area",ax=ax0,alpha=0.2)
    ax = ax0.twinx()
    STsPr['2000':'2003'][4000].plot(ax=ax, legend=False)

    CONC[5000].plot(kind="area",ax=ax1,alpha=0.2)
    axx = ax1.twinx()
    STsPr['2000':'2003'][5000].plot(ax=axx, legend=False)

    CONC[2000].plot(kind="area",ax=ax2,alpha=0.2)
    axxx = ax2.twinx()
    STsPr['2000':'2003'][2000].plot(ax=axxx, legend=False)
    plt.tight_layout()

    CONC[8000].plot(kind="area",ax=ax3,alpha=0.2)
    axxxx = ax3.twinx()
    STsPr['2000':'2003'][8000].plot(ax=axxxx, legend=False)
    plt.tight_layout()

    CONC[9000].plot(kind="area",ax=ax4,alpha=0.2)
    ax = ax4.twinx()
    STsPr['2000':'2003'][9000].plot(ax=ax, legend=False)

    CONC[7000].plot(kind="area",ax=ax5,alpha=0.2)
    axx = ax5.twinx()
    STsPr['2000':'2003'][7000].plot(ax=axx, legend=False)

    CONC[1000].plot(kind="area",ax=ax6,alpha=0.2)
    axxx = ax6.twinx()
    STsPr['2000':'2003'][1000].plot(ax=axxx, legend=False)
    plt.tight_layout()

    CONC[3000].plot(kind="area",ax=ax7,alpha=0.2)
    axxxx = ax7.twinx()
    STsPr['2000':'2003'][3000].plot(ax=axxxx, legend=False)
    plt.tight_layout()

    print(CONC.columns)

###################################################################################
####### NEW TESTS ON RUNING THE BT STRATEGY:
def run_stbsii(sPr, signal, comissions=True):

    fees = lambda q, p: max(1., abs(q) * p * 0.25/100.)

    ss = bt.Strategy('s0',[bt.algos.RunDaily(),
                            bt.algos.SelectWhere(signal==True),
                            #bt.algos.SelectThese(signal),
                            bt.algos.WeighEqually(),
                            bt.algos.Rebalance()])

    if comissions == True:
        test = bt.Backtest(ss, sPr, commissions=fees)
        res = bt.run(test)
    else:
        test = bt.Backtest(ss, sPr)#, initial_capital=100000)
        res = bt.run(test)

    return res
###################################################################################
def get_signal(cond1,days_ewma,cond_ewma,cond_sbsi,LTmBsi,STmBsi,STsBsi,sPr):

    # Obtaining signals
    good_to_go     = LTmBsi > cond1

    # moving average of the LTmbsi
    _ewma = pd.rolling_mean(STsBsi,days_ewma)
    ewma = (STsBsi-_ewma)

    # COND1)
    # - moments where the ewma > 0.05 LTsBsi
    _cond1 = (ewma > cond_ewma).copy()

    # COND2)
    _cond2 = (STsBsi > cond_sbsi).copy()

    # COMBINE COND)
    _cond = (_cond1 & _cond2).copy()

    # COND3) if mbsi > 0
    for i in range(len(_cond2.index)):
        if good_to_go.ix[i].any() == False:
            _cond.ix[i] = False
        else:
            continue

    # create a dataframe with STsPr index
    T = pd.DataFrame(sPr.index,index=[sPr.index]); T = T.drop(['time'],axis=1)

    # concatenar T com _cond
    CONC = pd.concat([T,_cond], axis=1, join='outer')

    # clean
    CONC = CONC.drop([CONC.columns[-1],CONC.columns[-1-1]],axis=1)
    CONC = CONC.fillna(False).copy()

    return CONC
###################################################################################
"""
I am having troubles with the indexes, solve this issue!
"""
###################################################################################
def _clean_data_4_running_tests(Y,SHRT_MBSI,SHRT_SBSI,LT_MBSI,LT_SBSI,sPr):

    # Set indexes to be the same
    sPr = sPr[LT_SBSI.index[0]:Y.index[-1]].copy()
    Y   = Y[LT_SBSI.index[0]:Y.index[-1]].copy()
    SHRT_SBSI = SHRT_SBSI[LT_SBSI.index[0]:Y.index[-1]].copy()
    SHRT_MBSI = SHRT_MBSI[LT_SBSI.index[0]:Y.index[-1]].copy()
    LT_MBSI   = LT_MBSI[LT_SBSI.index[0]:Y.index[-1]].copy()
    LT_SBSI   = LT_SBSI[LT_SBSI.index[0]:Y.index[-1]].copy()

    # tirar os dois last indexes
    SHRT_SBSI = SHRT_SBSI.drop([6000,116],axis=1)
    #SHRT_MBSI = SHRT_MBSI.drop([1,116],axis=1)
    #LT_MBSI   = LT_MBSI.drop([1,116],axis=1)
    LT_SBSI   = LT_SBSI.drop([6000,116],axis=1)

    # print
    print(np.shape(LT_MBSI))
    print(np.shape(LT_SBSI))
    print(np.shape(SHRT_MBSI))
    print(np.shape(SHRT_SBSI))

    return Y,SHRT_MBSI,SHRT_SBSI,LT_MBSI,LT_SBSI,sPr
###################################################################################
def calc_cagr(prices):
    """
    Calculates the CAGR (compound annual growth rate) for a given price series.
    Args:
        * prices (pandas.TimeSeries): A TimeSeries of prices.
    Returns:
        * float -- cagr.
    """
    start = prices.index[0]
    end = prices.index[-1]
    return (prices.ix[-1] / prices.ix[0]) ** (1 / year_frac(start, end)) - 1
###################################################################################
def to_drawdown_series(prices):
    """
    Calculates the drawdown series.
    This returns a series representing a drawdown.
    When the price is at all time highs, the drawdown
    is 0. However, when prices are below high water marks,
    the drawdown series = current / hwm - 1
    The max drawdown can be obtained by simply calling .min()
    on the result (since the drawdown series is negative)
    Method ignores all gaps of NaN's in the price series.
    Args:
        * prices (TimeSeries or DataFrame): Series of prices.
    """
    # make a copy so that we don't modify original data
    drawdown = prices.copy()

    # Fill NaN's with previous values
    drawdown = drawdown.fillna(method='ffill')

    # Ignore problems with NaN's in the beginning
    drawdown[np.isnan(drawdown)] = -np.Inf

    # Rolling maximum
    roll_max = np.maximum.accumulate(drawdown)
    drawdown = drawdown / roll_max - 1.
    return drawdown
###################################################################################
def year_frac(start, end):
    """
    Similar to excel's yearfrac function. Returns
    a year fraction between two dates (i.e. 1.53 years).
    Approximation using the average number of seconds
    in a year.
    Args:
        * start (datetime): start date
        * end (datetime): end date
    """
    if start > end:
        raise ValueError('start cannot be larger than end')

    # obviously not perfect but good enough
    return (end - start).total_seconds() / (31557600)
###################################################################################
def run_portfolio_composition(mBsi,sBsi,sPr,cond1,cond2,fillnawzero=True):
    sPr      = sPr.fillna(method='ffill').copy()
    ret      = sPr.diff().copy() # should I use pct_change() or diff() ?
    NN       = [] # number of stocks within the portfolio
    ret = ret*(1/9.)
    ret = ret.fillna(method='ffill')

    ## Run the strategy
    PORTF = []
    for T2 in sBsi.index:
        # if MBSI > 0:
        if mBsi.ix[T2].values[0] > cond1:
            try:
                offset = T2 + pd.DateOffset(days=1) # 1 day delay for getting good indexes
                X = ret_GoodSIndx(offset, sBsi)
            except:
                # function that returns good sectors at a given t2
                offset = T2 + pd.DateOffset(days=3) # 3 day delay for getting good indexes
                offset = T2
                X = ret_GoodSIndx(offset, sBsi)

            N = np.float(np.size(X)) # Number of stocks within portfolio
            if N == 0:
                PORTF.append(0. * ret.ix[T2][X])
            else:
                #PORTF.append( (1/N) * ret.ix[T2][X]) # weight assets (1/N)
                PORTF.append(ret.ix[T2][X])
            NN.append(N) # NUMBER OF ASSETS WITHIN THE PORTFOLIO
            

        elif mBsi.ix[T2].values[0] <= cond2:
            NN.append(0)
            X = ret_GoodSIndx(T2, sBsi)
            PORTF.append(0. * ret.ix[T2][X])

    ## CALCULATIONS
    PRT = pd.DataFrame(PORTF)
    PRT = pd.DataFrame(PRT.mean(axis=1))
    PRT[sBsi.index[0]:sBsi.index[365]] = np.NaN
    if fillnawzero is not False:
        PRT = PRT.fillna(0)
    else:
        pass
    return PRT, NN
###################################################################################
def max_dd(ser):
    max2here = pd.expanding_max(ser)
    dd2here = ser - max2here
    return dd2here.min()
###################################################################################
def calculate_performance_statistics(Y,PRT,NN,cumsum=True,plot=True):

    if cumsum  == True:
        s = pd.Series(PRT[0].cumsum())
    else:
        s = pd.Series(PRT[0])

    # calculate drawdown
    number_years = year_frac(s.index[0],s.index[-1])
    
    # calc CAGR
    CAGR = ((s+1)[-1] ** (1/25.) - 1) * 100

    # weights
    weights = pd.DataFrame(NN,index=PRT.index)

    if cumsum is not True:
        # SHARPE RATIO & OTHERS
        SR = sharpe_ratio_calc(PRT[0].cumsum())
        CAGR = (PRT[0]+1).calc_cagr()
        print('sharpe ratio = %s'%(SR))
        print('CAGR = %s'%CAGR)
    else:
        # SHARPE RATIO & OTHERS((_Port1.cumsum()+1)[-1] ** (1/25.) - 1) * 100
        SR = sharpe_ratio_calc(PRT[0].cumsum())
        print('sharpe ratio = %s'%(SR))
        print('CAGR = %s'%CAGR)
    # do the trick for calculating DD
    PRT['prices'] = PRT.columns[0].copy()

    if plot is not False:
        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        fig, host = plt.subplots(1,1,figsize=(10,5))
        fig.subplots_adjust(right=0.75)

        par1 = host.twinx()
        par2 = host.twinx()

        # Offset the right spine of par2.  The ticks and label have already been
        par2.spines["right"].set_position(("axes", 1.12))
        make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)

        # figure
        if cumsum  == True:
            toPlot = pd.DataFrame(PRT[0].cumsum()); toPlot.index = Y.index 
            toPlot.plot(linewidth=0.8,color='b',ax=par1,alpha=0.69)
            par1.legend('')
            (toPlot+1).to_drawdown_series().plot(kind="area",linewidth=0.8,ax=par2,alpha=0.35,color='b')
            Y.to_drawdown_series().plot(kind="area",linewidth=0.8,ax=par2,alpha=0.25,color='k')
            par2.legend('')
        else:
            PRT[0].plot(linewidth=0.8,color='b',ax=par1)
            PRT[0].to_drawdown_series().plot(linewidth=0.8,ax=par2,alpha=0.5,color='r')
        Y.plot(grid=True,linewidth=0.8,color='k',ax=host,legend=False)
        
        # print DD
        print('max. DD = %f.2'%(toPlot+1.).to_drawdown_series().min())
        
        # labeling
        host.set_xlabel('')
        host.set_ylabel(r'$S&P\, 500$')
        par1.set_ylabel(r'$P&L$')
        par2.set_ylabel(r'$DD$')
        
        # limits
        host.set_ylim([0,Y.max().values[0]])
        if PRT[0].cumsum().max() < 1000:
            par1.set_ylim([0,PRT[0].cumsum().max()])
        else:
            par1.set_ylim([0,PRT[0].max()])
        par2.set_ylim([-1.1,0])
        
        plt.tight_layout()

    return CAGR, 0, SR
###################################################################################
def get_moving_averages(days,LT_MBSI,LT_SBSI):
    ewma_lt_market = pd.rolling_mean(LT_MBSI,days)
    ewma_lt_sector = pd.rolling_mean(LT_SBSI,days)
    difference_ewma_ltmbsi = LT_MBSI - ewma_lt_market
    difference_ewma_ltsbsi = LT_SBSI - ewma_lt_sector

    return difference_ewma_ltmbsi.fillna(0), difference_ewma_ltsbsi.fillna(0)
###################################################################################
def run_portfolio_compositionII(mBsi,sBsi,sPr,cond1,cond2,difference_ewma_ltmbsi,difference_ewma_ltsbsi,fillnawzero=False):
    
    sPr      = sPr.fillna(method='ffill').copy()
    ret      = sPr.diff().copy() # should I use pct_change() or diff() ?
    NN       = [] # number of stocks within the portfolio
    ret      = (1/9.)*ret
    ret      = ret.fillna(method='ffill')

    ## Run the strategy
    PORTF = []

    for T2 in sBsi.index:
        if mBsi.ix[T2].values[0] >= 0:
            # if MBSI > 0:
            if difference_ewma_ltmbsi.ix[T2][0] > cond1:
                try:
                    offset = T2 + pd.DateOffset(days=1) # 1 day delay for getting good indexes
                    X = ret_GoodSIndxII(offset, sBsi, difference_ewma_ltsbsi, cond2)
                except:
                    offset = T2 + pd.DateOffset(days=3) # 3 day delay for getting good indexes
                    X = ret_GoodSIndxII(offset, sBsi, difference_ewma_ltsbsi, cond2)

                N = np.float(np.size(X)) # Number of stocks within portfolio
                if N == 0:
                    PORTF.append(0. * ret.ix[T2][X])
                else:
                    #PORTF.append( (1/N) * ret.ix[T2][X]) # weight assets (1/N)
                    PORTF.append(ret.ix[T2][X])
                NN.append(N)                             # NUMBER OF ASSETS WITHIN THE PORTFOLIO

            elif difference_ewma_ltmbsi.ix[T2][0] <= cond2:
                NN.append(0)
                X = ret_GoodSIndxII(T2, sBsi, difference_ewma_ltsbsi, cond2)
                PORTF.append(0. * ret.ix[T2][X])
        else:
            PORTF.append(0. * ret.ix[T2][X])
            NN.append(0. * ret.ix[T2][X])

    ## CALCULATIONS
    PRT = pd.DataFrame(PORTF)
    PRT = pd.DataFrame(PRT.mean(axis=1))
    PRT[sBsi.index[0]:sBsi.index[252]] = np.NaN
    if fillnawzero is not False:
        PRT = PRT.fillna(0)
    else:
        pass
    return PRT, NN

#######################################################################################################################
def ret_GoodSIndxII(T2, SDF, difference_ewma_ltsbsi,cond2):

    # return the Sectors with positive SBSI array(X)
    X = SDF.ix[T2]
    X = X[difference_ewma_ltsbsi.ix[T2] > cond2]
    X = np.array(X.axes[0][0:-1])

    return X
#######################################################################################################################
def return_port_turnover(PRT,NN):

    turnover_df = pd.DataFrame(NN)
    turnover_df.index = PRT[0].index
    turnover_df = abs(turnover_df.diff().copy())*0.05
    prt_mit_turnover = pd.DataFrame((PRT[0].cumsum()) - turnover_df[0])

    return prt_mit_turnover, turnover_df


#######################################################################################################################
def pos2pnl(price,position , ibTransactionCost=False ):
    """
    calculate pnl based on price and position
    Inputs:
    ---------
    price: series or dataframe of price
    position: number of shares at each time. Column names must be same as in price
    ibTransactionCost: use bundled Interactive Brokers transaction cost of 0.005$/share

    Returns a portfolio DataFrame
    """

    delta=position.diff()
    port = DataFrame(index=price.index)

    if isinstance(price,Series): # no need to sum along 1 for series
        port['cash'] = (-delta*price).cumsum()
        port['stock'] = (position*price)

    else: # dealing with DataFrame here
        port['cash'] = (-delta*price).sum(axis=1).cumsum()
        port['stock'] = (position*price).sum(axis=1)



    if ibTransactionCost:
        tc = -0.005*position.diff().abs() # basic transaction cost
        tc[(tc>-1) & (tc<0)] = -1  # everything under 1$ will be ceil'd to 1$
        if isinstance(price,DataFrame):
            tc = tc.sum(axis=1)
        port['tc'] = tc.cumsum()
    else:
        port['tc'] = 0.

    port['total'] = port['stock']+port['cash']+port['tc']



    return port
#######################################################################################################################
def herfindal_index(NN,PRT):
    H = pd.DataFrame(NN,index=[PRT.index])
    H = H**2
    return H
#######################################################################################################################
def calculate_turnover_yearly(NN,PRT):
    NDF = pd.DataFrame(abs(np.diff(NN)))
    NDF.index = PRT.index[0:-1]
    
    YEAR = pd.Series(np.arange(1975, 2015),dtype='str')
    TT = pd.DataFrame(NDF['1975'].cumsum()/np.size(NDF['1990'].cumsum()))
    
    for year in YEAR:
        T = NDF[str(year)].cumsum()/np.size(NDF[str(year)].cumsum())
        T = pd.concat([TT,T],axis=0)
        TT = T.copy()
    
    return TT
#######################################################################################################################
def plot_turnover_yearly(TT):
    fig, ax = plt.subplots(figsize=(7,4))
    TT.plot(color='k',linestyle='',
            linewidth=0.5,markerfacecolor='k',legend=False,ax=ax)
    ax.set_ylim([0,1])
    plt.tight_layout()
    
##########################################################################################
def run_strategy_1_new(mBsi,sBsi,sPr,cond1,cond2,difference_ewma_ltmbsi,difference_ewma_ltsbsi,fillnawzero=False):
    
    sPr      = sPr.fillna(method='ffill').copy()
    ret      = sPr.diff().copy() # should I use pct_change() or diff() ?
    NN       = [] # number of stocks within the portfolio
    ret      = (1/9.)*ret
    ret      = ret.fillna(method='ffill')

    ## Run the strategy
    PORTF = []

    for T2 in sBsi.index:
        
        if mBsi.ix[T2].values[0] >= 0.:
            if difference_ewma_ltmbsi.ix[T2][0] > cond1:
                X = ret_GoodSIndxII(T2, sBsi, difference_ewma_ltsbsi, cond2) # get good sectors at (t)
                N = np.float(np.size(X)) # Number of stocks within portfolio
                if N == 0:
                    PORTF.append(0. * ret.ix[T2][X])
                else:
                    try:
                        offset_returns = T2 + pd.DateOffset(days=1)
                        PORTF.append(ret.ix[offset_returns][X])
                    except:
                        offset_returns = T2 + pd.DateOffset(days=3)
                        PORTF.append(ret.ix[offset_returns][X])
                    NN.append(N)                            
            elif difference_ewma_ltmbsi.ix[T2][0] <= cond2:
                NN.append(0)
                X = ret_GoodSIndxII(T2, sBsi, difference_ewma_ltsbsi, cond2)
                PORTF.append(0. * ret.ix[T2][X])
        else:
            PORTF.append(0. * ret.ix[T2][X])
            NN.append(0. * ret.ix[T2][X])

    ## CALCULATIONS
    PRT = pd.DataFrame(PORTF)
    PRT = pd.DataFrame(PRT.mean(axis=1))
    PRT[sBsi.index[0]:sBsi.index[252]] = np.NaN
    if fillnawzero is not False:
        PRT = PRT.fillna(0)
    else:
        pass
    return PRT, NN
##########################################################################################
def sharpe_ratio_calc(ts):
    """ ts => time-series
        rf => risk-free rate
        PS> Use price series (not returns)
    """
    
    # Set risk-free rate
    rf = 0.025
    
    # clean incoming data
    ts = ts.diff()
    ts = ts.replace([np.inf, -np.inf], np.nan)
    ts = ts.fillna(0)
    
    # calculate sharpiro (anualized)
    mu  = ts.mean(axis=0)
    std = ts.std(axis=0)
    sr = (mu/std)*np.sqrt(365)
    
    return sr