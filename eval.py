import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, time, argparse, os, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tqdm
from scipy.stats import pearsonr


def loadData(dataPath='', dataType='binary'):
    data = np.load(dataPath)

    if dataType == 'binary':
        data = np.clip(data, 0, 1)
    
    data = np.rint(data)
    return data


train_data = loadData('proc_data/mimic/mimic1782_train.npy')
temp_data = loadData("EHRDiff.npy")

train_data_mean = np.mean(train_data, axis = 0)
temp_data_mean = np.mean(temp_data, axis = 0)

nzc = sum(temp_data_mean[i] > 0 for i in range(temp_data_mean.shape[0]))
corr = pearsonr(temp_data_mean, train_data_mean)

fig, ax = plt.subplots(figsize=(8, 6))
slope, intercept = np.polyfit(train_data_mean, temp_data_mean, 1)
fitted_values = [slope * i + intercept for i in train_data_mean]
identity_values = [1 * i + 0 for i in train_data_mean]

ax.plot(train_data_mean, fitted_values, 'b', alpha=0.5)
ax.plot(train_data_mean, identity_values, 'r', alpha=0.5)
ax.scatter(train_data_mean, temp_data_mean, alpha=0.3)
ax.set_title('corr: %.4f, none-zero columns: %d, slope: %.4f'%(corr[0], nzc, slope))
ax.set_xlabel('real')
ax.set_ylabel('generated')

fig.savefig('./{}.png'.format('EHRDiff'))
plt.close(fig)