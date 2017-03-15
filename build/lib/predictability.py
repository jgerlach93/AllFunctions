import bt
import ffn
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pylppl as lp
import sys

sys.path.append('/Users/demos/Documents/Python/ipy (work)/LPPLS - Sloppy/')
import sloppy_func as fsl
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import label_binarize

import itertools

#####################################################################
# ROC CURVE
#####################################################################

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
    for i in range(len(y_hat)):
        if y_actual[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
    for i in range(len(y_hat)):
        if y_actual[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return TP, FP, TN, FN


#####################################################################

def calc_statistics(y_actual, y_hat):
    TP, FP, TN, FN = perf_measure(y_actual, y_hat)

    tp_rate = TP / np.sum(y_actual)
    fp_rate = FP / (len(y_actual) - np.sum(y_actual))
    specificity = 1 - fp_rate
    acc = (TP + np.sum(y_actual)) / (np.sum(y_actual) + (len(y_actual) - np.sum(y_actual)))
    precision = TP / (TP + FP)

    return tp_rate, fp_rate, acc, precision, specificity


#####################################################################
# signals and others
#####################################################################
def get_signals(DF, signals, thr, check=False, lrg_scale=True, med_scale=False, shrt_scale=False):
    # New signals according to DIDIER
    # Threshold for the LPPLS CONFIDENCE INDICATOR
    # EX: = 0.6

    if lrg_scale == True:
        _signals = signals['lrg'][signals['lrg'] > thr].dropna().index
    elif med_scale == True:
        _signals = signals['med'][signals['med'] > thr].dropna().index
    elif shrt_scale == True:
        _signals = signals['shrt'][signals['shrt'] > thr].dropna().index

    if check is not False:
        DF.plot(color='k')
        [plt.axvline(_signals[i], color='r') for i in range(len(_signals))]
        plt.tight_layout()

        return _signals
    else:
        pass

    return _signals


#####################################################################
def get_targets(DF, thr2):
    # Set the threshold for the Drawdown of Price
    # EX: thr2 = -4

    ret = (DF.pct_change() * 100).fillna(0)
    targets = ret[ret < thr2].dropna()

    return targets


#####################################################################
def binarize_signals(_signals, DF):
    # pre-assign for signals
    S = []

    # get
    for t2 in DF.index:
        if t2 in _signals:
            S.append(1)
        else:
            S.append(0)

    SDF = pd.DataFrame(S, index=DF.index, columns=['signals'])
    return SDF


#####################################################################
def binarize_targets(_targets, DF):
    # pre-assign for targets
    _targets.columns = ['targets']
    TDF = pd.concat([DF, _targets], axis=1).fillna(0)  #### WRONG!
    TDF = TDF.drop(['P/D'], axis=1)

    for t2 in TDF.index:
        if TDF.ix[t2][0] < 0:
            TDF.ix[t2][0] = 1
        else:
            pass

    # The targets
    return TDF


#####################################################################
def do_it_all_4_fixed_thr(_signals, DF):
    # get targets across several thresholds
    thr2 = np.linspace(-1, -10, 10)  # Several threshold for the drawdown
    # if thr2 = -2 it means that our targets are moments in time where the percentual change of the time-series
    # is < thr2. In other words, moments where a DD < thr2 occurred.

    # Pre-assign
    RES_tp_rate, RES_fp_rate = [], []

    for _thr in thr2:
        _targets = get_targets(DF, _thr)  # Here we get the dates where this occured

        # binarize
        TDF = binarize_targets(_targets, DF)
        SDF = binarize_signals(_signals, DF)
        y_actual, y_hat = TDF['targets'], SDF['signals']

        # Statistics
        tp_rate, fp_rate, acc, precision, specificity = calc_statistics(y_actual, y_hat)

        # Appending res
        RES_tp_rate.append(tp_rate)
        RES_fp_rate.append(fp_rate)

    return RES_tp_rate, RES_fp_rate


#####################################################################

#####################################################################
def get_targets_block_at_threshold(thr, DF, spikes=False):
    # calculate epsilon DD
    edd = lp.epsilon_drawdowns(pd.Series(DF['P/D']),
                               window=60,
                               epsilon=thr,
                               fix_delta=True,
                               last_one=True)
    # organize
    eDD = pd.DataFrame(edd, index=DF.index)

    # pick period
    y1, y2 = DF.index[0], DF.index[-1]

    # initialise
    t2_neg = []
    t2_pos = []

    # Here we append t2s where -1 occurs
    for t2 in eDD['turning_pts'][y1:y2][eDD['turning_pts'][y1:y2] == -1].index:
        t2_neg.append(t2)

    # Here we append t2s where 1 occurs
    for t2 in eDD['turning_pts'][y1:y2][eDD['turning_pts'][y1:y2] == 1].index:
        t2_pos.append(t2)

    if spikes == False:
        # Pre-alocate
        RES = pd.DataFrame()

        for t in range(len(t2_pos)):
            if t == 1:
                res = eDD['turning_pts'][str(t2_neg[0])[0:10]:str(t2_pos[1])[0:10]]
                RES = pd.concat([RES, res], axis=0)
            else:
                res = eDD['turning_pts'][str(t2_neg[t - 1])[0:10]:str(t2_pos[t])[0:10]]
                RES = pd.concat([RES, res], axis=0)

        # SET EVERYTHIN TO 1
        RES[0] = 1

        return RES
    else:
        neg = pd.DataFrame(t2_neg)
        pos = pd.DataFrame(t2_neg)

        RES = pos.append(neg)
        RES.index = RES[0].values
        RES[0] = 1

        return RES


#####################################################################
def binarize_alles(RES, DF, _signals):
    # binarize
    TDF = binarize_targets(RES, DF)
    SDF = binarize_signals(_signals, DF)

    # res
    y_actual, y_hat = TDF['targets'], SDF['signals']

    return y_actual, y_hat


#####################################################################
def do_it_all_2(_signals, DF, spikes=False):
    # get targets across several thresholds of the DRAWDOWN!!!
    thr2 = np.linspace(2, 12, 10)  # Several threshold for the drawdown

    # Pre-assign
    RES_tp_rate, RES_fp_rate = [], []

    for _thr in thr2:
        # Get targets at a given threshold
        RES = get_targets_block_at_threshold(_thr, DF, spikes=spikes)

        # Binarize
        y_actual, y_hat = binarize_alles(RES, DF, _signals)

        # Statistics
        tp_rate, fp_rate, acc, precision, specificity = calc_statistics(y_actual, y_hat)

        # Appending res
        RES_tp_rate.append(tp_rate)
        RES_fp_rate.append(fp_rate)

    return RES_tp_rate, RES_fp_rate


#####################################################################
def calc_statistics_at_given_T_for_confidence_long(thr, DF, signals, spikes=False):
    # Get them babies
    _signals = get_signals(DF, signals, thr, check=False,
                           lrg_scale=True,
                           med_scale=False,
                           shrt_scale=False)

    # Calculate
    RES_tp_rate, RES_fp_rate = do_it_all_2(_signals, DF, spikes=spikes)

    return RES_tp_rate, RES_fp_rate


def calc_statistics_at_given_T_for_confidence_med(thr, DF, signals, spikes=False):
    # Get them babies
    _signals = get_signals(DF, signals, thr, check=False,
                           lrg_scale=False,
                           med_scale=True,
                           shrt_scale=False)

    # Calculate
    RES_tp_rate, RES_fp_rate = do_it_all_2(_signals, DF, spikes=spikes)

    return RES_tp_rate, RES_fp_rate


def calc_statistics_at_given_T_for_confidence_short(thr, DF, signals, spikes=False):
    # Get them babies
    _signals = get_signals(DF, signals, thr, check=False,
                           lrg_scale=False,
                           med_scale=False,
                           shrt_scale=True)

    # Calculate
    RES_tp_rate, RES_fp_rate = do_it_all_2(_signals, DF, spikes=spikes)

    return RES_tp_rate, RES_fp_rate


#####################################################################
#####################################################################
def fraction_for_a_given_threshold(negDD, _signals, DF):
    """
    negDD    -> comes from a given threshold of the epsilon DD and thus must be optimised.
    _signals -> threshold for the indicator.
    DF -> single column vector of prices.
    """

    # boolean vector of t2s that are under a drawdown \in _signals
    simple_test = [negDD.index[i] in _signals for i in range(len(negDD))]

    # Are of the DD
    lenDD = len(negDD)

    # simple fraction
    fra = np.sum(simple_test) / np.float(lenDD)

    # calculate area of the signal
    lenSig = len(_signals)

    # Calculate total number of t2s in the time-seris
    lenData = len(DF)

    # Fraction of time ocupied by the alarm at a given threshold
    alarmArea = np.float(lenSig) / np.float(lenData)

    # Fraction of missed targets
    fracMissedTargets = 1.0 - (np.sum(simple_test) / np.float(lenDD))

    return fra, alarmArea, fracMissedTargets


#####################################################################
def iterate_over_sev_thr(DF, negDD, signals, epsilon_thr, lrg_scale=True, med_scale=False, shrt_scale=False):
    Thr = np.linspace(0.0, 1.0, 20)
    _fra = []
    _alarmArea = []
    _missedTargets = []

    for i in range(len(Thr)):
        _signals = get_signals(DF, signals, Thr[i], check=False, lrg_scale=lrg_scale, med_scale=med_scale,
                               shrt_scale=shrt_scale)
        fra, alarmArea, fracMissedTargets = fraction_for_a_given_threshold(negDD, _signals, DF)
        _fra.append(fra)
        _alarmArea.append(alarmArea)
        _missedTargets.append(fracMissedTargets)

    return pd.DataFrame(_fra, index=Thr), pd.DataFrame(_alarmArea, index=Thr), pd.DataFrame(_missedTargets, index=Thr)


#####################################################################
def plot_results_triple_axis(simple_fraction, alarmArea, missedTargets, epsilon_thr, negDD, posDD):
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fs = 16
    ms = 10

    fig, host = plt.subplots(2, 1, figsize=(10, 7))

    # simple plot
    negDD.plot(color='r', linewidth=.8, ax=host[0])
    posDD.plot(color='g', linewidth=.8, ax=host[0])
    host[0].tick_params(axis='both', which='major', labelsize=16)
    host[0].tick_params(axis='both', which='minor', labelsize=16)
    host[0].legend('', loc='best')
    host[0].set_title('Epsilon DrawDown Threshold = %s' % epsilon_thr, fontsize=18)
    host[0].grid(True)

    fig.subplots_adjust(right=0.75)

    par1 = host[1].twinx()
    par2 = host[1].twinx()

    par2.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)

    simple_fraction.plot(marker='s', ax=host[1], label='', linewidth=2.5, markersize=ms)
    missedTargets.plot(ax=par1, color='r', marker='o', label='', linewidth=2.5, markersize=ms)
    alarmArea.plot(ax=par2, color='k', marker='D', label='', linewidth=2.5, markersize=ms)
    host[1].axhline(0.5, linewidth=2.5, linestyle='--', color='k')

    host[1].set_xlabel("LPPLS Confidence Indicator Threshold", fontsize=fs)
    host[1].set_ylabel("captured targets in %", fontsize=fs, color='b')
    par1.set_ylabel("missed targets in %", fontsize=fs, color='r')
    par2.set_ylabel("Alarm area in %", fontsize=fs)

    tkw = dict(size=4, width=1.5)
    host[1].tick_params(axis='y', colors='b', **tkw)
    par1.tick_params(axis='y', colors='r', **tkw)
    par2.tick_params(axis='y', colors='k', **tkw)
    host[1].tick_params(axis='x', **tkw)

    par1.grid(True)

    host[1].legend('', loc='best')
    par1.legend('', loc='best')
    par2.legend('', loc='best')
    host[1].set_ylim([0, 1]);
    par1.set_ylim([0, 1]);
    par2.set_ylim([0, 1])

    host[1].tick_params(axis='both', which='major', labelsize=16)
    host[1].tick_params(axis='both', which='minor', labelsize=16)
    par1.tick_params(axis='both', which='major', labelsize=16)
    par1.tick_params(axis='both', which='minor', labelsize=16)
    par2.tick_params(axis='both', which='major', labelsize=16)
    par2.tick_params(axis='both', which='minor', labelsize=16)

    plt.tight_layout()


#####################################################################
def get_targets(DF, thr_epsilon, signals, plot=False):
    if plot == False:
        # Calculate the epsilon drawdowns
        edd = lp.epsilon_drawdowns(pd.Series(DF[DF.columns[0]]),
                                   window=60,
                                   epsilon=thr_epsilon,
                                   fix_delta=True,
                                   last_one=True)

        # extract
        test = lp.extract_drawup_drawdown_series(edd)

        # positive and negative drawdowns
        posDD = pd.DataFrame(test[0], columns=['drawup'])
        negDD = pd.DataFrame(test[1], columns=['drawdown'])
        negDD.index = negDD.index.normalize()
        posDD.index = posDD.index.normalize()

        return posDD, negDD
    else:
        # Calculate the epsilon drawdowns
        edd = lp.epsilon_drawdowns(pd.Series(DF['P/D']),
                                   window=60,
                                   epsilon=thr_epsilon,
                                   fix_delta=True,
                                   last_one=True)

        # extract
        test = lp.extract_drawup_drawdown_series(edd)

        # positive and negative drawdowns
        posDD = pd.DataFrame(test[0], columns=['drawup'])
        negDD = pd.DataFrame(test[1], columns=['drawdown'])
        negDD.index = negDD.index.normalize()
        posDD.index = posDD.index.normalize()

        # Visualise
        f, ax = plt.subplots(1, 1, figsize=(16, 5))
        DF.plot(ax=ax, color='k', linewidth=2.)
        a = plt.twinx()
        signals['lrg'].plot(kind='area', color='r', linewidth=2., alpha=0.5, ax=a)
        negDD.plot(color='r', ax=ax, linewidth=2.)
        plt.grid(True)
        a.axhline(thr_epsilon, color='k', linewidth=2.5, linestyle='--')
        plt.tight_layout()

        return posDD, negDD


#####################################################################
def calculate_over_two_thresholds_and_one_time_scale(DF, signals, lrg_scale=True, med_scale=False, shrt_scale=False):
    import matplotlib.backends.backend_pdf

    # range of threshold for the epsilon DD
    epsilon_vec = np.linspace(1, 30, 10)

    pdf = matplotlib.backends.backend_pdf.PdfPages("/Users/demos/Desktop/test1.pdf")
    for i in range(len(epsilon_vec)):
        # getting them DD
        posDD, negDD = get_targets(DF, epsilon_vec[i], signals, plot=False)

        # calculate
        simple_fraction, alarmArea, missedTargets = iterate_over_sev_thr(DF['1983':'2016'],
                                                                         negDD['1983':'2016'],
                                                                         signals['1983':'2016'], epsilon_vec[i],
                                                                         lrg_scale=lrg_scale, med_scale=med_scale,
                                                                         shrt_scale=shrt_scale)

        # plot
        plot_results_triple_axis(simple_fraction, alarmArea, missedTargets, epsilon_vec[i], negDD, posDD)
        pdf.savefig()
    pdf.close()


#####################################################################
def binarize_a_df(df, thr):
    """
    return a dataframe = 1 for values > thr.
    and = 0 for values < 0.
    """
    return (df >= thr).astype(int)


#####################################################################
def get_final_df_for_logit_regression(signals, DF, y1, y2, thr_sig, debug=True, plot=False, lrg=True, med=False,
                                      shrt=False, return_quartiles_mean=False):
    """
    This function returns Y := quartiles of returns after each t2 (10 days after by default)
    and signals := discretized LPPLS confidence indicators at a given threshold (thr_sig)
    NB! -> Cuidado com o value do thr_sig since it changes depending on the time-scale im using (short, med or large)
    EX: y1, y2 = '1996','1998'
        -> HERE WE ARE USING Returns 10 days after each t2. We can change as we wish.
    """

    # Getting the .25 quantile of r_t for all t \in t2's.
    RES = fsl.get_cond_ret_dataframe(signals, DF[y1:y2], debug=debug)

    # Define the signal
    if med is not False:
        fdiffSig = pd.DataFrame((signals.med.pct_change() * 100).fillna(0))
    elif lrg is not False:
        fdiffSig = pd.DataFrame((signals.lrg.pct_change() * 100).fillna(0))
    elif shrt is not False:
        fdiffSig = pd.DataFrame((signals.shrt.pct_change() * 100).fillna(0))

    if plot is not False:
        f, ax = plt.subplots(2, 1, figsize=(7, 4))
        DF[y1:y2].plot(ax=ax[0])
        ax[0].set_ylabel('Price')
        a = ax[0].twinx()
        fdiffSig[y1:y2].plot(ax=a, color='r')
        ax[0].grid(True)
        RES[10].plot(ax=ax[1])
        ax[1].grid(True)
        plt.tight_layout()
    else:
        pass

    res = binarize_a_df(fdiffSig[y1:y2], thr_sig)
    # HERE WE ARE USING Returns 10 days after each t2. We can change as we wish. NB !!!
    TTT = pd.DataFrame(RES[10].values, index=res.index, columns=['quartiles'])
    TTT['signal'] = res.values
    TTT = TTT.dropna()

    if return_quartiles_mean==False:
        return TTT
    else:
        return TTT, res


#####################################################################
def scale_data(df):
    min_max_scaler = preprocessing.scale(df.fillna(0))

    return min_max_scaler


#####################################################################
def getting_boolean_vector_of_targets_based_on_epsilon_DD(DF, posDD, negDD):
    """
    :param DF:
    :param posDD:
    :param negDD:
    :return:
    """

    # new df for signals (drawups)
    pos_df = pd.DataFrame(index=DF.index.values)
    pos_df = pd.concat([pos_df, posDD])
    pos_df = pos_df.groupby(pos_df.index).first()
    pos_df = pos_df.fillna(0)
    pos_df[pos_df > 0] = 1
    pos_df.index = pd.DatetimeIndex(pos_df.index).normalize()

    # new df for signals (drawdowns)
    neg_df = pd.DataFrame(index=DF.index)
    neg_df = pd.concat([neg_df, negDD])
    neg_df = neg_df.groupby(neg_df.index).first()
    neg_df = neg_df.fillna(0)
    neg_df[neg_df > 0] = 1
    neg_df.index = pd.DatetimeIndex(neg_df.index).normalize()

    return pos_df, neg_df


#####################################################################
def get_binary_signal_at_threshold(DF, signal, thr_signal, shrt=False, med=False, lrg=True):
    # Define the signal
    if med is not False:
        sig_df = pd.DataFrame(index=DF.index)
        df = pd.DataFrame(signal['med'] >= thr_signal)
        sig_df = pd.concat([sig_df, df],axis=1)
    elif lrg is not False:
        sig_df = pd.DataFrame(index=DF.index)
        df = pd.DataFrame(signal['lrg'] >= thr_signal)
        sig_df = pd.concat([sig_df, df],axis=1)
    elif shrt is not False:
        sig_df = pd.DataFrame(index=DF.index)
        df = pd.DataFrame(signal['shrt'] >= thr_signal)
        sig_df = pd.concat([sig_df, df],axis=1)

    return sig_df

#####################################################################
def roc_curve_for_given_signal(DF, signals, up_target, shrt=False, med=False, lrg=True):

    # SET RANGE FOR THE INDICATOR THRESHOLD
    rang = np.linspace(0.1,0.99,12)

    # START
    ROC_AUC, FPR, TPR = [], [], []
    for i in rang:

        # get signal
        binary_sig = get_binary_signal_at_threshold(DF, signals, i, shrt=shrt, med=med, lrg=lrg)

        # compute stuff
        fpr, tpr, tj = roc_curve(binary_sig.values.astype(int), up_target.values.astype(int))
        roc_auc = auc(fpr, tpr)
        FPR.append(fpr)
        TPR.append(tpr)
        ROC_AUC.append(roc_auc)

    return FPR, TPR, ROC_AUC, rang

#####################################################################
def plot_ROC(FPR, TPR, ROC_AUC, rang, title, ax=None):

    marker = itertools.cycle(('D', '+', 'x', 'o', '*','v','^'))

    if ax is not None:
        for i in range(len(rang)):
            ax.plot(FPR[i],TPR[i],
                     marker=marker.next(),
                     label='ROC curve of LPPLS thr = %.1f (area = {%.2f})'%(rang[i], ROC_AUC[i]))
            #ax.hold(True)
            ax.grid(True)
            ax.set_ylim([-0.1,1.1]); ax.set_xlim([-0.1,1.1])
            ax.plot([0,1], [0,1], color='k', linestyle='--', marker='s')
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0.)
            #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            #ax.legend(loc='lower right')
            ax.set_ylabel('True Positive Rate', fontsize=18)
            ax.set_xlabel('False Positive Rate', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=16)
            plt.tight_layout()
    else:
         f,ax = plt.subplots(1,1,figsize=(12,8))
         for i in range(len(rang)):
            ax.hold(True)
            ax.plot(FPR[i],TPR[i],
                    marker=marker.next(),
                    label='ROC curve of LPPLS thr = %.2f (area = {%.2f})'%(rang[i], ROC_AUC[i]))
            #plt.hold(True)
            ax.grid(True)
            ax.set_ylim([-0.1,1.1]); plt.xlim([-0.1,1.1])
            plt.plot([0,1], [0,1], color='k', linestyle='--', marker='s')
            #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax.legend(loc='lower right')
            ax.set_ylabel('True Positive Rate', fontsize=18)
            ax.set_xlabel('False Positive Rate', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=16)
            ax.set_title('Threshold Epsilon Negative DD = %s'%title,fontsize=18)

#####################################################################
def getting_ROC_metric(DF, signals, drawup=False, drawdown=True, shrt=False, med=False, lrg=True):

    ROC_large = []

    # ITERATION PLOT
    thr_epsilon = np.linspace(1,60,60)

    for i in range(len(thr_epsilon)):

        # GETTING TARGETS
        posDD, negDD = get_targets(DF, thr_epsilon[i], signals, plot=False)

        # Binarize targets
        # GETTING TARGETS (BINARY)
        up_target, down_target = getting_boolean_vector_of_targets_based_on_epsilon_DD(DF, posDD, negDD)

        if drawup == True:
            FPR, TPR, ROC_AUC, rang = roc_curve_for_given_signal(DF, signals, up_target, shrt=shrt, med=med, lrg=lrg)
            ROC_large.append(ROC_AUC)
        elif drawdown == True:
            FPR, TPR, ROC_AUC, rang = roc_curve_for_given_signal(DF, signals, down_target, shrt=shrt, med=med, lrg=lrg)
            ROC_large.append(ROC_AUC)

    # Create DF with results for any time scale and either pos or neg targets
    scl=np.linspace(0.1,0.99,12)
    ROC_DF = pd.DataFrame(ROC_large, index= thr_epsilon, columns=scl)

    return ROC_DF

#####################################################################
def plot_ROCArea_as_function_of_threshold_epsilon(ROC_LARGE,ROC_MED,ROC_SHRT):
    # Initialize the plot
    marker = itertools.cycle(('D', 'o', 'v'))
    f,ax = plt.subplots(3,1,figsize=(14,14),sharex=True)
    f.subplots_adjust(hspace=0.008)

    # figures
    ROC_LARGE.plot(ax=ax[0],marker=marker.next(),linewidth=2.5,markersize=10)
    ROC_MED.plot(ax=ax[1],marker=marker.next(),linewidth=2.5,markersize=10)
    ROC_SHRT.plot(ax=ax[2],marker=marker.next(),linewidth=2.5,markersize=10)

    # pormenores
    [ax[i].tick_params(axis='both', which='major', labelsize=16) for i in range(3)]
    [ax[i].tick_params(axis='both', which='minor', labelsize=16) for i in range(3)]
    [ax[i].set_ylabel('ROC (area under the curve)', fontsize=16) for i in range(3)]
    [ax[i].grid() for i in range(3)]
    [ax[i].set_ylim([0,1]) for i in range(3)]
    ax[0].set_title('Large-scale Indicator - target: drawdown/drawup', fontsize=16)
    ax[1].set_title('Medium-scale Indicator - target: drawdown/drawup', fontsize=16)
    ax[2].set_title('Small-scale Indicator - target: drawdown/drawup', fontsize=16)
    ax[2].set_xlabel('Epsilon Drawdown/Drawup Threshold', fontsize=16)

    #
    [plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=70, horizontalalignment='right' ) for i in [0,1,2]]
    plt.tight_layout()