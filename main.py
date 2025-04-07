import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# specify src
src="192.168.74.10"

# specify threshold percentile
th_perc = 10

# print alerts table
flag = 1

filename = "./datasciense/results/{}_processed_data_src.csv".format(src)
new_df = pd.read_csv(filename, index_col=0, parse_dates=True)


print(new_df.head())

print(type(new_df.index[0]))

### Split train, predict sets: time range: 2011-07-22, 2014-03-24
cut = datetime(2013, 7, 1).date()     # 1st day to predict
loc = new_df.index.slice_locs(end=cut)
train_df = new_df.iloc[loc[0]:loc[1], :]
predict_df = new_df.iloc[loc[1]:, :]


# X = new_df.copy()
X = train_df.copy()
qt_transformer = QuantileTransformer(output_distribution='normal')
qt_transformer.fit(X)
X_qt = qt_transformer.transform(X)
X_qt = pd.DataFrame(data=X_qt, index=X.index, columns=X.columns)

X_qt.shape

# Plot distribution after transform
# X_plot = X_qt
# plt.rcParams.update({'font.size': 6})
# figure, axis = plt.subplots(5, 4)
# for i in range(len(X_plot.columns)):
#     X_plot[X_plot.columns[i]].plot(ax=axis[i//4,i%4], kind='hist', bins=100, title=X_plot.columns[i])
# plt.tight_layout()
# file_name = "./datasciense/results/{}_distribution_after_transform_plot.svg".format(src)
# plt.savefig(file_name)
# plt.show()

### Compute anomaly score using probability

# compute probability
stats = X_qt.describe()

probs = pd.DataFrame(index=X_qt.index)
for x in X_qt.columns:
    mu, sigma = stats.loc['mean', x], stats.loc['std', x]
    for i in X_qt.index:
        probs.loc[i, x] = norm.sf(X_qt.loc[i, x], loc=mu, scale=sigma)
        # if X_qt.loc[i, x] <= mu: 
        #     probs.loc[i, x] = norm.cdf(X_qt.loc[i, x], loc=mu, scale=sigma)
        # else:
        #     probs.loc[i, x] = norm.sf(X_qt.loc[i, x], loc=mu, scale=sigma)

# compute log probability
log_probs = pd.DataFrame(index=probs.index)
for x in probs.columns:
    log_probs[x] = probs[x].apply(lambda x: math.log(x))

# compute anomaly score = sum of log probabilities
log_probs['score'] = log_probs.sum(axis=1)

# compute threshold score
th = np.percentile(log_probs['score'], th_perc)

### Prediction

# transform, compute probability, log probability, anomaly score
def probab_score(x, qt_transformer, stats):
    x_tran = qt_transformer.transform(x)
    x_tran = pd.DataFrame(data=x_tran, index=x.index, columns=x.columns)
    x_prob = pd.DataFrame(index=x_tran.index)
    x_log_prob = pd.DataFrame(index=x_tran.index)
    for x in x_tran.columns:
        mu, sigma = stats.loc['mean', x], stats.loc['std', x]
        for i in x_tran.index:
            x_prob.loc[i, x] = norm.sf(x_tran.loc[i, x], loc=mu, scale=sigma)
            # if x_tran.loc[i, x] <= mu: 
            #     x_prob.loc[i, x] = norm.cdf(x_tran.loc[i, x], loc=mu, scale=sigma)
            # else:
            #     x_prob.loc[i, x] = norm.sf(x_tran.loc[i, x], loc=mu, scale=sigma)
        x_log_prob[x] = x_prob[x].apply(lambda x: math.log(x))
        
    x_log_prob['score'] = x_log_prob.sum(axis=1)
    
    return x_tran, x_prob, x_log_prob

# x = new_df.copy()
# x = predict_df.copy()
x = train_df.copy()
x_tran, x_prob, x_log_prob = probab_score(x, qt_transformer, stats)

def alert_tb(src, x, x_tran, x_prob, x_log_prob, th, th_perc, qt_transformer, stats):
        # th, th_perc: threshold, threshold percentile
        
        mu_std = qt_transformer.inverse_transform(stats.loc[['mean', 'std'], :])
        mu_std = pd.DataFrame(data=mu_std, index=['mean', 'std'], columns=stats.columns)

        
        tb_list = []     # list of tables, one for each anomaly
        for i in x.index:
            # print(i)
            tbi = pd.DataFrame(index=['data', 'mean', 'std', 'transformed data', 
                                      'transformed mean', 'transformed std', 'probability'])
            for row in tbi.index[:1]:
                tbi.loc[row, 'src'] = src
                tbi.loc[row, 'date'] = i
            
                tbi.loc[row, 'score'] = f"{x_log_prob.loc[i, 'score']:.2f}"
                tbi.loc[row, 'score_th'] = f"{th:.2f}"
                tbi.loc[row, 'th_perc'] = th_perc
            
            for col in x.columns:
                # print(col)
                tbi.loc['data', col] = f"{x.loc[i, col]:.2f}"
                tbi.loc['data', "{}_↑_%".format(col)] = np.nan
                tbi.loc['data', "{}_↑abs_value".format(col)] = np.nan
                if mu_std.loc['mean', col] != 0:
                    tbi.loc['data', "{}_↑_%".format(col)] = "{}%".format(round((x.loc[i, col] - mu_std.loc['mean', col]) / mu_std.loc['mean', col] * 100, 2))
                else:
                    tbi.loc['data', "{}_↑abs_value".format(col)] = f"{abs(x.loc[i, col] - mu_std.loc['mean', col]):.2f}"
                tbi.loc['mean', col] = f"{mu_std.loc['mean', col]:.2f}"
                tbi.loc['std', col] = f"{mu_std.loc['std', col]:.2f}"
                tbi.loc['transformed data', col] = f"{x_tran.loc[i, col]:.2f}"
                tbi.loc['transformed mean', col] = f"{stats.loc['mean', col]:.2f}"
                tbi.loc['transformed std', col] = f"{stats.loc['std', col]:.2f}"
                tbi.loc['probability', col] = f"{x_prob.loc[i, col]:.2f}"
            
            tb_list.append(tbi)
        tb = pd.concat(tb_list)
        return tb

anom_ind = x_log_prob[x_log_prob['score'] < th].index
if not anom_ind.empty:
    tb = alert_tb(src, x.loc[anom_ind, :], x_tran.loc[anom_ind, :], x_prob.loc[anom_ind, :], \
                  x_log_prob.loc[anom_ind, :], th, th_perc, qt_transformer, stats)
else:
    tb = "No anomaly found."

if isinstance(tb, pd.DataFrame) and not tb.empty:
    file_name = "./datasciense/results/{}_anomely_tb_src_train_1112.csv".format(src)
    tb.to_csv(file_name)
else:
    print(tb)


if flag == 0:
    # predict
    x_line = pd.concat([pd.Series(log_probs.index), pd.Series(x_log_prob.index)])
    # th_line = [th] * len(log_probs)
    th_line = [th] * len(x_line)
    plt.xlabel('date')  
    plt.ylabel('anomaly score')
    plt.xticks(rotation=30)
    plt.plot(log_probs.index, log_probs['score'], 'b.', label='Training data')
    # plt.plot(log_probs.index, th_line, 'r', label='Threshold: 10 percentile')
    plt.plot(x_log_prob.index, x_log_prob['score'], 'g.', label='Prediction data')
    plt.plot(x_line, th_line, 'r', label='Threshold: 10 percentile')
    plt.legend(loc='lower left')
    
    file_name = "./datasciense/results/{}_anomely_plot_1112.svg".format(src)
    plt.savefig(file_name)
    
    plt.show()

if flag == 1:
    # train
    # plot anomaly score with threshol line
    x_line = pd.concat([pd.Series(log_probs.index), pd.Series(x_log_prob.index)])
    th_line = [th] * len(log_probs)
    # th_line = [th] * len(x_line)
    plt.xlabel('date')
    plt.ylabel('anomaly score')
    plt.xticks(rotation=30)
    plt.plot(log_probs.index, log_probs['score'], 'b.')
    # plt.plot(log_probs.index, log_probs['score'], 'b.', label='Training data')
    plt.plot(log_probs.index, th_line, 'r', label='Threshold: 10 percentile')
    # plt.plot(x_log_prob.index, x_log_prob['score'], 'g.', label='Prediction data')
    # plt.plot(x_line, th_line, 'r', label='Threshold: 10 percentile')
    plt.legend(loc='lower right') # , plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0.0))

    file_name = "./datasciense/results/{}_anomely_plot_train_1112.svg".format(src)
    plt.savefig(file_name)

    plt.show()

