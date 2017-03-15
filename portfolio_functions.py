__author__ = 'demos'

import pandas as pd
import bt
import pylppl as lp
from pydatastream import Datastream
import sklearn.decomposition as dec
from sklearn import preprocessing
import itertools
import sys
import strategy_func as sdfn
import portfolio_functions as pf
from pandas_datareader import data, wb
import datetime
import sloppy_func as fsl
import token, tokenize
from pandas_datareader import data, wb
from six.moves import cStringIO as StringIO
import json
from pandas.io.json import json_normalize
import urllib2 as ul
import requests

import data_functions as dfn

import bt as bt
from ffn import *

import matplotlib as mp
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 17}

label_size = 15
mp.rcParams['xtick.labelsize'] = label_size
mp.rcParams['ytick.labelsize'] = label_size
try:
    import talib as TA
    import talib
    import cvxopt as opt
    from cvxopt import blas, solvers
except:
    print('no talib')
    pass

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

######

"""
-----------------------
FUNDAMENTAL INDICATORS:
-----------------------

- DY: The dividend yield expresses the dividend per share as a percentage of the share price. The underlying dividend is calculated according to the same principles as datatype DPSC (Dividend per share, current rate) in that it is based on an anticipated annual dividend and excludes special or once-off dividends. Dividend yield is calculated on gross dividends (including tax credits) where available. Note that dividend yield for UK, Irish and French stocks is calculated on gross dividends (including tax credits), although dividends per share for these countries are displayed net.
- MV: Market value on Datastream is the share price multiplied by the number of ordinary shares in issue. The amount in issue is updated whenever new tranches of stock are issued or after a capital change. For companies with more than one class of equity capital, the market value is expressed according to the individual issue. Market value is displayed in millions of units of local currency.
- PE: This is the price divided by the earnings rate per share at the required date. For full details of the price and earnings figures used in any particular case, see the Price and Earnings per share topics.
    - There maybe no such thing as a good price-to-earnings ratio.
      When P/E is high, one can either say its too expensive or argue that growth prospects are good.
      On the other hand, when P/E is low, one can say that it is a value play or that the company's future is not too bright.
- DPS: It is intended to represent the anticipated payment over the following 12 months and for that reason may be calculated on a rolling 12-month basis, or as the "indicated" annual amount
- PA: Price ASK
- PB: Price BID
- VWAP - Volume Weighted Average Price: Volume Weighted Average Price (VWAP) is calculated by dividing the total volume of shares traded for a stock on a particular day, into the total value of shares traded for the stock on that day.
- WC03019 - Deposits Total represent the value of money held by the bank or financial company on behalf of its customers. The item includes demand, savings, money market and certificates of deposit along with foreign office and deposit accounts.
- DWED  - EBITDA
- PTBV - Price 2 book value - ~ any value under 1.0 is considered a good P/B value ~
- NOSH  - Number of shares
- DWCX - Capital expenditure
- DWNP - Net profit income
- MTBV - This is defined as the market value of the ordinary (common) equity divided by the balance sheet value of the ordinary (common) equity in the company
- WC05474 - Closely held shares: CLOSELY HELD SHARES - CURRENT represents shares held by insiders. For companies with more than one class of common stock, closely held shares for each class is added together.
- WC06023 - City where company is located
- WC06092 - Company description
- WC18272 - Date company founded
- WC07211 - Market Capitalization Current (U.S.$)
- WC05091 - Market Price 52 Week High
- BTAC - Beta Correlation: This figure measures how closely changes in the price of a particular share will follow changes in the level of the market as a whole.
- ECUR - Current earnings
- W09502 - Dividend Payout Per Share - Current (Security)
- WC08372 - Return On Equity - Per Share - Current
- WC01001 - Net sales or revenues
- WC02201 - Total assets
- WC03101 - Current Liabilities Total
- WC03351 - Total Liabilities

--------------------------------------
----------- > WHAT IVE LEARNED so far:
--------------------------------------

    - I entered the stock market in the wrong time on september 06 2016.
        Must create an indicator to enter the stock market and fight the feeling of losing something
        if I dont enter the market as soon as possible. That is a recipe for failure.

    - Must pre-select stocks not only based on beta or past performance
        Must rank based on several fundamental indicators and backtest.

    - I should've shorted BRZ VS USD ON JANUARY !!
        I KNEW IT

    - Gold always surge on JANUARY ! (buy gold in January)

    - Metric for picking stocks with the highest upward acceleration probability
      and then construt short-term strategies for it with lots of capital alocated.

"""

#######################################################################################################
# Library for developing trading strategies #
#
# - Remember that stock picking is the most important task
# - Then we move to the correct weightning
# - We also cover all fundamental indicator analysis
# -
# -
########################################################################################################

########################################################################################################
def update_ibovespa(DWE, res, last_observation_date, today):
    # Loop for getting all stocks from IBOVESPA
    DF = pd.DataFrame()
    for asset in res.IBTKR:
        try:
            df = DWE.get_price(asset, date_from=last_observation_date, date_to=today)
            DF[asset] = df['P']
        except:
            print('error retrieving asset %s'%asset)
            DF[asset] = 0
    return DF

########################################################################################################
def df_with_fundamental_indicator(DWE, i, f_ind, st, en, static=False):
    """
        i => ticker
        f_ind => fundamental indicator as string

        EXAMPLE:
        f_ind = 'DY'
        i = '@:@SBV'
        df_with_fundamental_indicator(i,f_ind,st,en)

        ##### ##### ##### ##### ##### ##### #####

        # Example for fetching static data
        f_ind = 'WC06023'
        i = '@:@SBV'
        st = '2016-01-01'
        en = '2016-02-01'

        df_with_fundamental_indicator(i, f_ind, st, en, static=True)

    """
    if static == False:
        x = DWE.fetch([i],fields=[f_ind],date_from=st, date_to=en,freq='D')
        return x
    else:
        x = [DWE.fetch([i],fields=[f_ind],static=True)]
        return x

########################################################################################################
def iterate_and_append(DWE, DF, f_ind, st, en):

    """ Iterate over all stocks from a dataframe with ticker """

    X=[]
    for i in DF.ticker:
        try:
            x = df_with_fundamental_indicator(DWE, i, f_ind, st, en)
            X.append(x)
        except:
            X.append(0)

    return X

########################################################################################################
def iterate_and_append_static(DWE, DF, f_ind, st, en):

    """ Iterate over all stocks from a dataframe with ticker """

    X=[]
    for i in DF.ticker:
        try:
            x = df_with_fundamental_indicator(DWE, i, f_ind, st, en, static=True)
            X.append(x[0][f_ind][0])
        except:
            X.append(0)

    return X

########################################################################################################
def get_all_fundamental_indicator_all_stocks(res, DWE, st, en):
    # create a list of fundamentals
    fund_list = ['WC03101','WC02201','WC01001','WC08372','W09502','ECUR',
                 'BTAC','WC05091','WC07211','WC18272','WC06092','WC06023','WC05474',
                 'MTBV','DWNP','NOSH','PTBV','DWED','WC03019','VWAP',
                 'DPS','PE','MV','DY','EPS','WC02201','WC03351','INDM2','INDM3','INDM4','INDM5']

    # CREATE A DATAFRAME
    cols = [i for i in res['NAME']]
    DF = pd.DataFrame(cols,columns=['company'])
    DF['sector'] = res.INDM
    DF['ticker'] = res.MNEM

    # Iterate over all fundamental indicators and create a dataframe DF
    for f_ind in fund_list:
        print(f_ind)
        #tmp = iterate_and_append_static(DF,f_ind,st,en) # Use for static data fetching
        tmp = iterate_and_append(DWE, DF, f_ind, st, en) # Use for time-series data fetching
        DF[f_ind] = tmp

    return DF

########################################################################################################
def get_updated_df(username,password,DWE):
    # Stablish connection with reuters
    #DWE = Datastream(username=username, password=password)

    # Today is?
    today = str(pd.datetime.today().date())

    # Fecth the ibovespa index
    Ybov = DWE.fetch(['BRBOVES'],date_from='2005-01-01',date_to=today)

    # get ibovespa constituents
    res = DWE.get_constituents('BRBOVES')

    # load all time series WITCH TICKER AS COLUMNS
    DF_tcker = pd.read_hdf('/Users/demos/Documents/Python/ipy (work)/Portfolio optimization - bt package/ibovespa_data_2010_2015_ticker.h5','res')
    last_observation_date = str(DF_tcker.index[-1])[0:10]

    DF_tcker_2 = update_ibovespa(DWE, res, last_observation_date, today)
    DF_tcker_2.columns = res.NAME # Set stock names as Columns
    new_df  = pd.concat([DF_tcker,DF_tcker_2])

    # save new dataframe to be loaded afterwards
    new_df.to_hdf('/Users/demos/Documents/Python/ipy (work)/Portfolio optimization - bt package/ibovespa_data_2010_2015_ticker.h5','res')

    # Clean duplicated entries
    new_df = new_df.groupby(new_df.index).first()

    # Check
    print_checkup(new_df,Ybov,today)

    return Ybov, res, new_df

########################################################################################################

no_func = lambda self: True
fees = lambda q, p: abs(q) * p * 0.25/100.

########################################################################################################
def run_strategy(fweight, data, N_m_dt, flimit=no_func, num_assets=40, fees=True):

    # RunMonthly eh o bixo

    # number of months for lookback (ie. 12, 3, 6, ..)
    m_dt = pd.DateOffset(months=N_m_dt)

    st = bt.Strategy('s0', [bt.algos.RunQuarterly(),
                            bt.algos.SelectHasData(lookback=m_dt),
                            bt.algos.SelectMomentum(num_assets, lookback=m_dt,
                                                    lag=pd.DateOffset(days=1),
                                                    all_or_none=True),
                            fweight,
                            flimit,
                            bt.algos.Rebalance()])

    if fees is not True:
        test = bt.Backtest(st, data,
                           integer_positions=True)
    else:
        fees = lambda q, p: abs(q) * p * 0.25/100.
        test = bt.Backtest(st, data,  initial_capital=10000.,
                           commissions=fees,
                           integer_positions=True)

    res = bt.run(test)

    return res

########################################################################################################
def normalize_data_for_comparacy(DF, sklearn=False):

    # Interpolate missing values
    DF = DF.fillna(0).copy()

    if sklearn == True:
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(DF)

        # pd.DataFrame
        DF = pd.DataFrame(X_train_minmax,index=DF.index,columns=DF.columns)

        return DF.copy()

    else:
        # We use the rebase function from ffn
        DF = DF.rebase()

        return DF.copy()

########################################################################################################
def calculate_EWMA(data):

    TICK = data.columns
    if len(TICK)>1:
        TICK = 'Close'
    else:
        pass

    today = str(pd.datetime.today().date())

    # calc_vol
    vol = pd.rolling_var(data,30)
    days_shrt, days_long = 100, 300

    #calculate
    shrt_mean = pd.rolling_mean(data[TICK],days_shrt)
    long_term = pd.rolling_mean(data[TICK],days_long)

    return shrt_mean, long_term, vol

########################################################################################################
def plot_cross_moving_average(data):

    """pass a dataframe containing prices only"""

    shrt_mean, long_term, vol = calculate_EWMA(data)

    TICK = data.columns
    if len(TICK)>1:
        TICK = 'Close'
    else:
        pass

    #plot
    f,ax = plt.subplots(1,1,figsize=(12,5))
    data[TICK].plot(color='k', ax=ax)
    ax.set_yticklabels([''])
    ax.grid()
    b = ax.twinx()
    b.set_yticklabels([''])
    shrt_mean.plot(ax=b, linewidth=3)
    b.legend('', loc='upper left')
    ax.legend('', loc='upper left')
    a = ax.twinx()
    a.set_yticklabels([''])
    long_term.plot(color='r', ax=a, linewidth=3)
    a.legend('', loc='upper left')
    b = a.twinx()
    b.set_yticklabels([''])
    vol.plot(kind="area", color='r', alpha=0.3, ax=b)
    plt.tight_layout()
    ax.set_xlabel('')
    b.legend('', loc='upper left')

    #for i in range(len(shrt_mean.index[shrt_mean > long_term])):
        #ax.axvline(shrt_mean.index[shrt_mean > long_term][i],color='b',alpha=0.1)
    plt.tight_layout()

def print_checkup(DF,Y,today):
    print('Today is %s, last DF is %s and last Y is %s'%(today, DF.index[-1], Y.index[-1]))

########################################################################################################
def plot_all_sectors(res,DF,sklearn=False):

    # list all sectors
    #sectors = res.INDC.unique()
    sectors = res.INDM.unique()

    # normalize data
    DF_N = normalize_data_for_comparacy(DF,sklearn=sklearn)

    # I skip one sector for ploting purposes
    f,ax = plt.subplots(7,4,figsize=(15,15))
    axs = ax.ravel()
    for sec in range(len(sectors[0:-1])):
        try:
            for i in res['NAME'][res.INDM == np.str(sectors[0:-1][sec])].values:
                DF_N[str(i)].plot(ax=axs[sec])
                axs[sec].set_title('sec: %s'%sectors[0:-1][sec])
                axs[sec].grid(True)
        except:
            pass
    plt.tight_layout()
    
########################################################################################################
def returnListOfStocksFromASector(res, selectedSec):
    lst = res[res['INDM']==selectedSec].NAME.values
    return lst[:]


def getStocksFromSector(res, DF, sectors2invest, single=False, i=0):
        
    if single == False:
        SS = pd.DataFrame()

        for i in range(len(sectors2invest)):
            ls = returnListOfStocksFromASector(res, sectors2invest[i])
            SS = pd.concat([SS, DF[ls]], axis=1)

        return SS
    else:
        ls = returnListOfStocksFromASector(res, sectors2invest[i])
        selectedStocks = DF[ls]

        return selectedStocks

########################################################################################################
def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

########################################################################################################
def random_portfolio(returns):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

########################################################################################################
def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])

    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']

    return np.asarray(wt), returns, risks

########################################################################################################
def get_all_sectors_DF(res,DF):
    """return all sectors as mean of all stocks within"""

    sectors_unique = res.INDM.unique()

    SEC = pd.DataFrame()

    for sec in sectors_unique:
        # Get the stocks from that index
        resu = res.ix[res.INDM == str(sec)].NAME.values

        # len of results
        number_stocks_on_sec = len(resu)

        if number_stocks_on_sec > 1:
            # Get the price time-series for all of em
            mean_stocks = pd.DataFrame(DF[resu[0:-1]].mean(axis=1), columns=[str(sec)])
            SEC = pd.concat([SEC,mean_stocks],axis=1)
        else:
            mean_stocks = DF[resu[0]]
            SEC = pd.concat([SEC,mean_stocks],axis=1)

    return SEC

########################################################################################################
def calc_beta(DF, Y):

    """
    \beta = Covariance (ri,rm )/Variance of Market
    """

    DF = DF.copy()
    DF = (DF.to_log_returns()*100).fillna(0)

    Y = Y.copy()
    Y = (Y.to_log_returns()*100).fillna(0)

    # PRE
    stocks = DF.columns
    cov = []

    # PRE I
    X = [np.cov(DF[str(stocks[i])], Y[Y.columns[0]].values) for i in range(len(stocks))]
    cov = [X[i][0][1] for i in range(len(stocks))]

    # PRE II
    beta = [cov[i] / np.var(Y)[0] for i in range(len(stocks))]

    # DATA FRAME
    beta = pd.DataFrame(beta,index=stocks).T

    return beta

########################################################################################################
def ponder_returns_by_beta(beta,DF,Y):

    stocks = DF.columns

    beta_pondered_time_series = []

    for i in range(len(stocks)):
        beta_pondered_time_series.append(DF[stocks[i]].values / beta[stocks[i]][0])

    beta = pd.DataFrame(beta_pondered_time_series).T
    beta.index = DF.index
    beta.columns = stocks

    return beta

########################################################################################################
"""
                        FUNCTIONS FOR MONITORING PORTFOLIOS
"""
########################################################################################################

def return_statistics_for_portfolio1(DF):
    oPrice_brf = 52.64                           # Old price of the asset
    nShare_brf = 10.                             # Number of shares
    nPrice_brf = DF['BRF BRASIL FOODS ON'][-1]   # New price of the asset

    oPrice_abv = 18.90
    nShare_abv = 20.
    nPrice_abv = DF['AMBEV ON'][-1]

    Total_port_at_t0 = (oPrice_brf * nShare_brf) + (oPrice_abv * nShare_abv)
    Total_port_today = (nPrice_brf * nShare_brf) + (nPrice_abv * nShare_abv)

    # Today is:
    today = str(pd.datetime.today().date())

    # Create the portfolio dataframe (with statistics)
    port_df = pd.DataFrame(Total_port_at_t0, columns=['port_total_t0'], index=[today])
    port_df['port_total_t1'] = Total_port_today

    # Calculus:
        # - 1) simple variation from t0 to t1
        # - 2) return in RS

    port_df['variation'] = ((port_df['port_total_t1'] - port_df['port_total_t0']) / port_df['port_total_t0']) * 100
    port_df['ret. in RS'] = port_df['port_total_t1'] - port_df['port_total_t0']

    return port_df

########################################################################################################

def monitoring_portfolio1(DF,Y):
    """
    plotting DD's
    drawdown from t0 to t1 > -5% ?
    """

    # TODAY
    today = str(pd.datetime.today().date())
    print('Today is %s'%today)

    ## CHECKING DROP IN THE TIME-SERIES
    rets = DF.pct_change()*100
    rets_ibov = Y.pct_change()*100

    var_ambev = rets['AMBEV ON'][-1]
    var_brf   = rets['BRF BRASIL FOODS ON'][-1]
    var_ibovespa = rets_ibov['P'][-1]

    print('Last P variation: %s to %s'%(DF['AMBEV ON'].index[-1-1], DF['AMBEV ON'].index[-1]))
    print('AMBEV3 = %.2f | BRFS3 = %.2f | IBOV = %.2f '%(var_ambev, var_brf, var_ibovespa))

    DDS = sdfn.to_drawdown_series(DF)

    f,ax = plt.subplots(1,2,figsize=(11,4))
    DDS['BRF BRASIL FOODS ON'].plot(ax=ax[0], color='r', kind='area', alpha=0.3)
    a = ax[0].twinx()
    DF['BRF BRASIL FOODS ON'].plot(color='k',ax=a)
    ax[0].set_title('BRF BRASIL FOODS ON')
    ax[0].grid()

    DDS['AMBEV ON'].plot(ax=ax[1], color='r', kind='area', alpha=0.3)
    a = ax[1].twinx()
    DF['AMBEV ON'].plot(color='k',ax=a)
    ax[1].set_title('AMBEV ON')
    ax[1].grid()

    plt.tight_layout()

########################################################################################################

def add_index_to_talib_ind(indi, dataframe):
    return pd.DataFrame(indi, index=dataframe.index)

########################################################################################################

def plot_port_statistics_monitoring(DF, Y):
    # port
    port = (DF['BRF BRASIL FOODS ON']['20-07-2016':] + DF['AMBEV ON']['20-07-2016':])/2.
    port = pd.DataFrame((port/port[0])*100)

    f, ax = plt.subplots(1,2, figsize=(13,3.5))
    ax[0].set_title(r'BRF & AMBV')
    DF['BRF BRASIL FOODS ON']['20-07-2016':].plot(ax=ax[0], marker='o')
    ax[0].legend(loc='best')
    a = ax[0].twinx()
    DF['AMBEV ON']['20-07-2016':].plot(color='g', marker='s')
    ax[0].grid()
    port.plot(ax=ax[1], color='k', marker='D')
    ((Y['20-07-2016':] / Y.ix['20-07-2016'] )*100).plot(ax=ax[1], color='r', marker='s')
    ax[1].legend(['portfolio','IBovespa'], loc='best')
    ax[1].grid()

    # Statistics
    mean_retu = (port.values[-1]/port.values[0])
    print('Sharpe ratio =>  %.3f'%(mean_retu / np.std(port)))
    print('Return made  =>  %.3f percent'%((mean_retu-1) * 100))
    print('Money made   =>  %.3f RS'%(((mean_retu-1)) * 904.4))
    print('Total Money  =>  %.3f RS'%(((mean_retu-1)) * 904.4 + 904.4))


########################################################################################################

def compare_pnl_two(r1, r2, logy=False):

    if logy == False:
        r1.backtests['s0'].strategy.data.value.plot(color='r', linestyle='-')
        r2.backtests['s0'].strategy.data.value.plot(color='k', linestyle='-')
        plt.ylabel(r'$P&L$', fontsize=22)
        plt.legend(['r1','r2'], fontsize=18, loc='best')
        plt.grid()
        plt.tight_layout()
    else:
        r1.backtests['s0'].strategy.data.value.plot(logy=True, color='r', linestyle='-')
        r2.backtests['s0'].strategy.data.value.plot(logy=True, color='k', linestyle='-')
        plt.ylabel(r'$log(P&L)$', fontsize=22)
        plt.legend(['r1','r2'], fontsize=18, loc='upper left')
        plt.grid()
        #plt.ylim([10**3.9, 10**4.8])
        plt.tight_layout()

########################################################################################################

def plot_port2_statistics_monitoring(DF, Y):

    stocks = ['BRF BRASIL FOODS ON',
           'ENERGIAS DO BRASIL ON BRAZIL','ENGIE BRASIL ENERGIA ON','EQUATORIAL ENERGIA ON',
           'ESTACIO PARTICIPACOES ON','NATURA COSMETICOS ON','QUALICORP ON',
           'RAIA DROGASIL ON','SMILES ON','TELEFONICA BRASIL PN','ULTRAPAR PARTICIPOES ON',
           'EMBRAER ON']

    weights = np.array([7., 14., 10., 8., 23., 12., 36., 6., 14., 8., 5., 44.])
    buying_p = np.array([55.15, 14.57, 42.30, 51.20, 17.70, 32.40, 22.40,
                         62.93, 52.40, 50.15, 73.00, 16.00])

    start = '09-06-2016'
    mv = 2

    # portfolio
    stocks_price_data_frame = DF[stocks][start:]

    port_t0 = []

    # Evolution
    for i in range(len(stocks)):
        port_t0.append(buying_p[i] * weights[i])

    # PNL of port_t0
    pnl_t0 = np.sum(port_t0)
    pnl_t1_stocks = stocks_price_data_frame * weights

    # plot
    mr = ['s','o','D','x','v','^','d','8','1','H','>','<'] * 2
    marker = itertools.cycle(('s','o','D','x','v','^','d','8','1','H','>','<'))

    f, ax = plt.subplots(3,1,figsize=(13, 15), sharex=True)
    [(DF[stocks[i]][start:]/DF[stocks[i]][start]).plot(ax=ax[0], marker=mr[i], linewidth=1.5,
                                                       markersize=11,
                                                       markevery=mv) for i in range(len(stocks))]
    ax[0].axhline([1.], color='k', linestyle='-', linewidth=4)
    ax[0].set_ylabel('stocks in $\%$: 09-06-16', fontsize=22)
    ax[0].grid(linewidth=.4, linestyle='-')
    pnl_t1_stocks.plot(marker=marker.next(),ax=ax[1], linewidth=4)
    ax[0].legend(stocks, numpoints = 1, loc='upper center', bbox_to_anchor=(0.46, 1.35),
                 shadow=True, ncol=4, fontsize=11)
    ax[1].grid(linewidth=.4, linestyle='-')
    ax[1].set_ylabel(r'Stock position in R\$', fontsize=22)
    ax[1].legend('')
    pnl_t1_stocks.sum(axis=1).plot(ax=ax[2], color='k',linestyle='-',linewidth=1, marker='o',
                                   markersize=12, markevery=mv)
    ax[2].set_ylim([pnl_t1_stocks.sum(axis=1).min()-500, pnl_t1_stocks.sum(axis=1).max()+500])
    #ax[2].grid(linewidth=1)
    ax[2].set_ylabel('Portfolio value in RS', fontsize=22)
    a = ax[2].twinx()
    ((pnl_t1_stocks.sum(axis=1) / pnl_t1_stocks.sum(axis=1)[0])*100).plot(ax=a, linewidth=2, linestyle='-', marker='s',
                                                                          markersize=12, markevery=mv)
    ((Y[start:] / Y.ix[start] )*100).plot(ax=a, color='r', marker='D', linewidth=2, markersize=12, markevery=mv)
    a.legend(['portfolio pnl','IBovespa DD'], fontsize=14, loc='upper center')

    a.set_ylabel(r'$P&L\, (blue)$', fontsize=22)
    a.axhline([100.], color='k', linestyle='-', linewidth=4)
    a.set_ylim([((pnl_t1_stocks.sum(axis=1) / pnl_t1_stocks.sum(axis=1)[0])*100).min()-15,
                ((pnl_t1_stocks.sum(axis=1) / pnl_t1_stocks.sum(axis=1)[0])*100).max()+15])
    a.grid(linewidth=.4, linestyle='-')
    plt.tight_layout()
    plt.subplots_adjust(hspace = .05)

    retMade = (pnl_t1_stocks.sum(axis=1).ix[-1] - pnl_t1_stocks.sum(axis=1).ix[0])/pnl_t1_stocks.sum(axis=1).ix[0] * 100

    print('money made %s RS' %(pnl_t1_stocks.sum(axis=1).ix[-1] - pnl_t1_stocks.sum(axis=1).ix[0]))
    print('return: %.2f percent'%(retMade))

    return pnl_t0.round(), pnl_t1_stocks.round()

########################################################################################################

def fetch_ohlc_data(start, end, ticker="^BVSP"):

    return web.DataReader(ticker, 'yahoo', start, end)

########################################################################################################

def plot_oscilator_ohlc_vs_index(df_ohlc, y1, y2):

    """Show oscillator for entering the market"""

    # IF THOMSON REUTERS
    try:# Get the oscillator
        r = TA.PLUS_DI(df_ohlc['PH'].values, df_ohlc['PL'].values, df_ohlc['P'].values)
        R = pd.DataFrame(r, index=df_ohlc.index)
        change_regime_ind = TA.HT_TRENDMODE(df_ohlc['P'].values)
        CR = pd.DataFrame(change_regime_ind, index=df_ohlc.index)
    except:
        r = TA.PLUS_DI(df_ohlc['High'].values, df_ohlc['Low'].values, df_ohlc['Close'].values)
        R = pd.DataFrame(r, index=df_ohlc.index)
        change_regime_ind = TA.HT_TRENDMODE(df_ohlc['Close'].values)
        CR = pd.DataFrame(change_regime_ind, index=df_ohlc.index)

    # Calciulate first and second moment
    df_ohlc = df_ohlc.fillna(method='backfill')
    stt   = np.std(R[y1:y2])
    m_ind = np.mean(R[y1:y2].dropna())[0]

    # Plot
    f,ax = plt.subplots(2,1, figsize=(12,8))
    try:
        df_ohlc[['Close','High','Low']][y1:y2].plot(ax=ax[0], linewidth=2)
    except:
        df_ohlc['P'][y1:y2].plot(ax=ax[0], linewidth=2)
    ax[0].grid()
    a = ax[0].twinx()
    R[y1:y2].plot(ax=a, color='k', linewidth=.8)
    a.axhline(m_ind, color='k', linewidth=3)
    a.axhspan(m_ind-stt, m_ind+stt, color='r', alpha=0.2)
    a.legend('', loc='lower left')
    aaa = ax[0].twinx()
    aaa.set_yticklabels('')
    CR[y1:y2].plot(ax=aaa, color='k', linewidth=.5)
    aaa.legend('', loc='lower left')

    # plot the moving average on top
    shrt_ewma, long_ewma, _ = calculate_EWMA(R[y1:y2])
    R[y1:y2].plot(ax=ax[1], color='k', linewidth=.8)
    ax[1].grid()
    ax[1].axhline(m_ind, color='k', linewidth=3)
    ax[1].axhspan(m_ind-stt, m_ind+stt, color='r', alpha=0.2)
    ax[1].legend('', loc='lower center')
    aa = ax[1].twinx()
    shrt_ewma.plot(ax=aa, color='b', linewidth=3)
    long_ewma.plot(ax=aa, color='r', linewidth=3)
    ax[1].legend(['long_EWMA','shrt_ewma'], fontsize=14, loc='lower left')

    plt.tight_layout()

########################################################################################################

def plot_ohlc_ind_vs_sp500(y1,y2):

    """
    Plot oscilators (OHLC) from TALIB for the SP500 and IBOV with std and mean
    """

    # Today
    today = str(pd.datetime.today().date())

    # Fetch IBOV
    ibov_df = pf.fetch_ohlc_data('2010', today, ticker="^BVSP")
    SP_df = pf.fetch_ohlc_data('2010', today, ticker="^GSPC")

    # plot
    pf.plot_oscilator_ohlc_vs_index(ibov_df, y1, y2)
    pf.plot_oscilator_ohlc_vs_index(SP_df, y1, y2)

########################################################################################################

# FUNDAMENTAL VALUATION

########################################################################################################
def getLowAndHighBetaStocksDf(DF, Y):

    # GET BETA
    beta = pf.calc_beta(DF, Y)

    # Create rankling
    beta_ranking = (beta.T).sort(columns=0)

    # Get returns of the lower 30/60 beta stocks (LOW BETA)
    returns_low_beta = beta_ranking[1:30]

    # Get returns of the upper 30/60 beta stocks (HIGH BETA)
    returns_high_beta = beta_ranking[30:-1]

    # HIGH BETA STOCKS DF
    high_beta_stocks = DF[returns_high_beta.index].copy()

    # LOW BETA STOCKS DF
    low_beta_stocks = DF[returns_low_beta.index].copy()

    return low_beta_stocks, high_beta_stocks


def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))


def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))


def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))


def pickGrowthStocks(data, n):
    """
    1) High PE and
    2) High P2B
    data -> FI_DF
    n -> number of stocks to pick

    We return only stocks \in the intersection of conditions 1) and 2)
    """

    highPe = data['PE'].sort_values(ascending=False).dropna()[0:n]
    highP2b = data['PTBV'].sort_values(ascending=False).dropna()[0:n]
    stocks = intersect(highPe.index, highP2b.index)
    gStocks = [str(stocks[i]) for i in range(len(stocks))]

    return gStocks


def pickValueStocks(data, n):
    """
    1) High Return On Equity - Per Share - Current (WC08372)
    2) LOW PE
    data -> FI_DF
    n -> number of stocks to pick

    We return only stocks \in the intersection of conditions 1) and 2)
    """

    highPe = data['PE'].sort_values(ascending=True).dropna()[0:n]
    highP2b = data['WC08372'].sort_values(ascending=False).dropna()[0:n]
    stocks = intersect(highPe.index, highP2b.index)
    vStocks = [str(stocks[i]) for i in range(len(stocks))]

    return vStocks


def marketCapOverNetSales(data, n):
    metric = (data['WC07211']/data['WC01001']).sort_values(ascending=False).dropna()[0:n]
    metric = [str(metric[i]) for i in range(len(metric))]
    return metric.index


def threeStrategiesAllocation(DF,Y,gStocks,vStocks):

    # VALUE STOCKS
    m_dt = pd.DateOffset(months=6)
    num_assets = 2

    # Strategy 1
    mom_s = bt.Strategy('mom_s', [bt.algos.RunQuarterly(),
                                  bt.algos.SelectHasData(lookback=m_dt),
                                  bt.algos.SelectMomentum(num_assets,
                                                         lag=pd.DateOffset(days=1),
                                                         all_or_none=True),
                                  bt.algos.WeighEqually(),
                                  bt.algos.Rebalance()],
                        [vStocks[0],vStocks[1],vStocks[2],vStocks[3]])

    # GROWTH STOCKS
    m_dt = pd.DateOffset(months=6)
    num_assets = 5
    st1 = bt.Strategy('s1', [bt.algos.RunQuarterly(),
                                bt.algos.SelectHasData(lookback=m_dt),
                                bt.algos.SelectMomentum(num_assets, lookback=m_dt,
                                                        lag=pd.DateOffset(days=1),
                                                        all_or_none=True),
                                bt.algos.WeighEqually(),
                                bt.algos.Rebalance()],
                      [gStocks[0],gStocks[1],gStocks[2],gStocks[3],gStocks[4],gStocks[5],gStocks[6],gStocks[7]])

    # DF
    m_dt = pd.DateOffset(months=6)
    num_assets = 1
    st2 = bt.Strategy('s2', [bt.algos.RunQuarterly(),
                                bt.algos.SelectHasData(lookback=m_dt),
                                bt.algos.SelectMomentum(num_assets, lookback=m_dt,
                                                        lag=pd.DateOffset(days=1),
                                                        all_or_none=True),
                                bt.algos.WeighEqually(),
                                bt.algos.Rebalance()])


    # create the master strategy - this is the top-most node in the tree
    # Once again, we are also specifying  the children. In this case, one of the
    # children is a Security and the other is a Strategy.
    master = bt.Strategy('s0', [bt.algos.RunQuarterly(),
                                    bt.algos.SelectAll(),
                                    bt.algos.WeighEqually(),
                                    bt.algos.Rebalance()],
                        [mom_s, st1, st2])

    # create the backtest and run it
    fees = lambda q, p: abs(q) * p * .25/100.
    t = bt.Backtest(master, DF, initial_capital=10000.,
                   commissions=fees, integer_positions=True)
    r = bt.run(t)

    # plot
    sdfn.plot_res(r,DF,Y)

    return r


def plotLowVsHighBetaAtYeari(lowBeta,highBeta,year,retAx=True):
    pf.normalize_data_for_comparacy(lowBeta[year].mean(axis=1)).plot(ax=retAx, color='b', linewidth=2)
    pf.normalize_data_for_comparacy(highBeta[year].mean(axis=1)).plot(ax=retAx, color='r', linewidth=2)
    retAx.grid()
    retAx.legend(['Low beta stocks ($\mu$)','High beta stocks ($\mu$)'], fontsize=11,
               loc='upper left', fancybox=True, framealpha=0.2)
    retAx.set_ylabel(r'$ln[P(t)]$', fontsize=22)
    plt.tight_layout()


def obtainingLpplsIndicatorAtT2(t2, data, job_pool=None):
    # Set dt range
    dt_range = np.linspace(30, 750, 120)

    # Run in parallel
    if RUN_PARALLEL:
        if job_pool is None:
            job_pool = Parallel(n_jobs=CPU_COUNT)
            res = job_pool(delayed(fsl.estimate_lppls)(data, dt, t2) for dt in dt_range)

    # Return dataframe containing parameters
    _res = pd.DataFrame()
    for i in range(len(res)):
        _res = pd.concat([_res,res[i]], axis = 0)

    return _res


def monitoringPortfolio3(DF, Y, normed=False):

    t1 = '2016-11-09'
    lst = ['VALE ON', 'EMBRAER ON', 'COMPANHIA SIDERURGICA NACIONAL ON']
    
    lw = 3

    w = np.array([37., 55., 90.])

    dataNnormed = DF[lst][t1:]

    if normed==False:
        data = DF[lst][t1:]
        y = Y[t1:]
    else:
        data = normalize_data_for_comparacy(DF[lst][t1:])-100
        y = normalize_data_for_comparacy(Y[t1:])-100

    port = dataNnormed[lst] * w
    port = port.sum(axis=1)

    # Money made in rs
    r = port.ix[-1] - port.ix[0]

    # Mean ret
    mret = data.mean(axis=1)

    # Plot
    f,ax = plt.subplots(1,3,figsize=(15,4))
    data.plot(ax=ax[0], linewidth=lw)
    ax[0].legend(['vale','embr','csn'], bbox_to_anchor=(.5, 1.05))
    ax[0].set_ylabel('ret in %', fontsize=15)
    data.mean(axis=1).plot(ax=ax[1], marker='s',linewidth=lw)
    ax[1].set_ylabel('$\mu$(ret.) in %', fontsize=15)
    y.plot(ax=ax[1], color='r', marker='o',linewidth=lw)
    ax[1].legend(['strtg.','ibvsp'], bbox_to_anchor=(.5, 1.05))
    port.plot(ax=ax[2], marker='D',linewidth=lw)
    ax[2].set_ylabel('portf. ret. in $', fontsize=15)
    ax[2].legend(['RS = %s \n ret in pp: %.2f '%(r,mret.ix[-1]) ], bbox_to_anchor=(.8, 1.05),
                 framealpha=0.2)
    plt.tight_layout()


    print('money made: %s RS'%r)
    print('return of port.: %.3f in pp'%mret.ix[-1])

    return data


###########################################################################
##### WORKING WITH OPTIONS
###########################################################################

def fix_lazy_json(in_text):
    """
    Handle lazy JSON - to fix expecting property name
    this function fixes the json output from google
    http://stackoverflow.com/questions/4033633/handling-lazy-json-in-python-expecting-property-name
    """
    tokengen = tokenize.generate_tokens(StringIO(in_text).readline)

    result = []
    for tokid, tokval, _, _, _ in tokengen:
        # fix unquoted strings
        if (tokid == token.NAME):
            if tokval not in ['true', 'false', 'null', '-Infinity', 'Infinity', 'NaN']:
                tokid = token.STRING
                tokval = u'"%s"' % tokval

        # fix single-quoted strings
        elif (tokid == token.STRING):
            if tokval.startswith ("'"):
                tokval = u'"%s"' % tokval[1:-1].replace ('"', '\\"')

        # remove invalid commas
        elif (tokid == token.OP) and ((tokval == '}') or (tokval == ']')):
            if (len(result) > 0) and (result[-1][1] == ','):
                result.pop()

        # fix single-quoted strings
        elif (tokid == token.STRING):
            if tokval.startswith ("'"):
                tokval = u'"%s"' % tokval[1:-1].replace ('"', '\\"')

        result.append((tokid, tokval))

    return tokenize.untokenize(result)

def Options(symbol):
    url = "https://www.google.com/finance/option_chain"
    r = requests.get(url, params={"q": symbol,"output": "json"})
    content_json = r.text
    dat = json.loads(fix_lazy_json(content_json))
    puts = json_normalize(dat['puts'])
    calls = json_normalize(dat['calls'])
    np=len(puts)
    nc=len(calls)

    for i in dat['expirations'][1:]:
        r = requests.get(url, params={"q": symbol,"expd":i['d'],"expm":i['m'],"expy":i['y'],"output": "json"})
        content_json = r.text
        idat = json.loads(fix_lazy_json(content_json))
        puts1 = json_normalize(idat['puts'])
        calls1 = json_normalize(idat['calls'])
        puts1.index = [np+i for i in puts1.index]
        calls1.index = [nc+i for i in calls1.index]
        np+=len(puts1)
        nc+=len(calls1)
        puts = puts.append(puts1)
        calls = calls.append(calls1)
    calls.columns = ['Ask','Bid','Chg','cid','PctChg','cs','IsNonstandard','Expiry','Underlying','Open_Int','Last','Symbol','Strike','Vol']
    puts.columns = ['Ask','Bid','Chg','cid','PctChg','cs','IsNonstandard','Expiry','Underlying','Open_Int','Last','Symbol','Strike','Vol']
    calls['Type'] = ['call' for i in range(len(calls))]
    puts['Type'] = ['put' for i in range(len(puts))]
    puts.index = [i+len(calls) for i in puts.index]
    opt=pd.concat([calls,puts])
    opt['Underlying']=[symbol for i in range(len(opt))]
    opt['Underlying_Price'] = [dat['underlying_price'] for i in range(len(opt))]
    opt['Root']=opt['Underlying']
    for j in ['Vol','Strike','Last','Bid','Ask','Chg']:
        opt[j] = pd.to_numeric(opt[j],errors='coerce')
    opt['IsNonstandard']=opt['IsNonstandard'].apply(lambda x:x!='OPRA')
    opt = opt.sort_values(by=['Strike','Type'])
    opt.index = range(len(opt))
    col = ['Strike', 'Expiry', 'Type', 'Symbol', 'Last', 'Bid', 'Ask', 'Chg', 'PctChg', 'Vol', 'Open_Int', 'Root', 'IsNonstandard', 'Underlying', 'Underlying_Price', 'cid','cs']
    opt = opt[col]

    return opt
