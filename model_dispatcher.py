from math import sqrt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from scipy.stats import boxcox
import traceback
import logging
import pandas as pd
import pmdarima as pm
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from matplotlib import colors
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os

def convertIndexToDateDF(filename):
    col_names = ["date", "value"]
    df = pd.read_csv(filename,
                     names = col_names, header = 0, parse_dates = [0])
    df['date'] = pd.to_datetime(df['date'],infer_datetime_format=True)
    df = df.set_index(['date'])
    return df

def checkStationary(data):
    val = adfuller(data)
    if (val[1] < 0.05):
        return True
    else:
        return False

def makeStationary(dataframe):
    diff = np.diff(dataframe)
    return diff

def getP_Q(data, difference):
    p_range = list(range(1,8))  # taking values from 1 to 4
    q_range = list(range(1,8))  # taking values from 1 to 8
    aic_values = []
    pq_values = []
    d = difference
    for p in p_range:
        for q in q_range:
            try:
                model = sm.tsa.arima.ARIMA(data, order=(p, d, q),enforce_stationarity=False)
                results = model.fit()
                aic_values.append(results.aic)
                pq_values.append((p, q))
            except Exception as e:
                logging.error(traceback.format_exc())


    best_pq = pq_values[aic_values.index(min(aic_values))]  # (p,q) corresponding to lowest AIC score
    print("(p,q) corresponding to lowest AIC score: ", best_pq)
    return best_pq[0], best_pq[1]

def modelArima(data,p,d,q):
    arima_model = sm.tsa.arima.ARIMA(data, order=(p, d, q),enforce_stationarity=False).fit()
    return arima_model

def model_Autoarima(data):
    m12 = pm.auto_arima(data, error_action='ignore', seasonal=True, m=12)
    return m12

