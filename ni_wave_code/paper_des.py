# ___ SET UP
# ______________________________________________________________________________________________________________
# NAMESPACES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.helpers_talent import *

# DATA
group = 4           # age class 'AK'
test_size = 0.2     # train-test split
dat, X_0, y_0, _, _, _, _ = load_data(group=group, test_size=test_size, sub=True) # subjectives = True
dat = dat[['TZP', 'Geburtstag_Taggenau', 'Grösse', 'Gewicht', 'SL20', 'GW',
 'DR', 'BK', 'BJ', 'SKSC_TAK', 'SKSC_TEC', 'SKSC_KON', 'SKSC_PSY',
 'Subj_Aktuelle_Leistungsfähigkeit', 'Subj_Zukünftiges_Leistungsniveau',
 'LZ']]
# ______________________________________________________________________________________________________________

# ___ DESCRIBTIVES
print('The data contains {0} players and {1} variables'.format(dat.shape[0], dat.shape[1]))
dat[dat['LZ']==1].shape

# describe
dat.describe()
dat[['Geburtstag_Taggenau', 'Grösse', 'Gewicht']].describe()
dat[['SL20', 'GW', 'DR', 'BK', 'BJ']].describe()
dat[['SKSC_TAK', 'SKSC_TEC', 'SKSC_KON', 'SKSC_PSY']].describe()
dat[['Subj_Aktuelle_Leistungsfähigkeit', 'Subj_Zukünftiges_Leistungsniveau']].describe()

# ___ TESTS
succ =  dat[dat['LZ']==1]
fail = dat[dat['LZ']==0]
# multivariate t-test (Hotelling's T^2)
import pingouin as pg
pg.multivariate_ttest(succ,fail)
# t-tests
from scipy.stats import ttest_ind
for name in dat.columns[1:-3]:
     stat, p = ttest_ind(succ[name],fail[name])
     print('For ', name, 'the test statistic is %.3f and p=%.3f' % (stat, p))

# success
succ[['Geburtstag_Taggenau', 'Grösse', 'Gewicht']].describe()
succ[['SL20', 'GW', 'DR', 'BK', 'BJ']].describe()
succ[['SKSC_TAK', 'SKSC_TEC', 'SKSC_KON', 'SKSC_PSY']].describe()
succ[['Subj_Aktuelle_Leistungsfähigkeit', 'Subj_Zukünftiges_Leistungsniveau']].describe()
# fail
fail[['Geburtstag_Taggenau', 'Grösse', 'Gewicht']].describe()
fail[['SL20', 'GW', 'DR', 'BK', 'BJ']].describe()
fail[['SKSC_TAK', 'SKSC_TEC', 'SKSC_KON', 'SKSC_PSY']].describe()
fail[['Subj_Aktuelle_Leistungsfähigkeit', 'Subj_Zukünftiges_Leistungsniveau']].describe()

# ___ PLOT
fig, axs = plt.subplots(2, 4)
bool_means = True
axs[0,0].boxplot([succ['Geburtstag_Taggenau'],fail['Geburtstag_Taggenau']],0,'',showmeans=bool_means, meanline=True)
axs[0,0].set_title('Geburtstag_Taggenau')
axs[0,0].set_xticklabels(['selected','non'],rotation=45, fontsize=8)

axs[0,1].boxplot([succ['Grösse'],fail['Grösse']],0,'',showmeans=bool_means, meanline=True)
axs[0,1].set_title('Grösse')
axs[0,1].set_xticklabels(['selected','non'],rotation=45, fontsize=8)

axs[0,2].boxplot([succ['Gewicht'],fail['Gewicht']],0,'',showmeans=bool_means, meanline=True)
axs[0,2].set_title('Gewicht')
axs[0,2].set_xticklabels(['selected','non'],rotation=45, fontsize=8)

axs[0,3].boxplot([succ['SL20'],fail['SL20']],0,'',showmeans=bool_means, meanline=True)
axs[0,3].set_title('SL20')
axs[0,3].set_xticklabels(['selected','non'],rotation=45, fontsize=8)

axs[1,0].boxplot([succ['GW'],fail['GW']],0,'',showmeans=bool_means, meanline=True)
axs[1,0].set_title('GW')
axs[1,0].set_xticklabels(['selected','non'],rotation=45, fontsize=8)

axs[1,1].boxplot([succ['DR'],fail['DR']],0,'',showmeans=bool_means, meanline=True)
axs[1,1].set_title('DR')
axs[1,1].set_xticklabels(['selected','non'],rotation=45, fontsize=8)

axs[1,2].boxplot([succ['BK'],fail['BK']],0,'',showmeans=bool_means, meanline=True)
axs[1,2].set_title('BK')
axs[1,2].set_xticklabels(['selected','non'],rotation=45, fontsize=8)

axs[1,3].boxplot([succ['BJ'],fail['BJ']],0,'',showmeans=bool_means, meanline=True)
axs[1,3].set_title('BJ')
axs[1,3].set_xticklabels(['selected','non'],rotation=45, fontsize=8)

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.2, wspace=0.3)