import model_dispatcher as md
import pandas as pd
import numpy as np
# import config
import configparser
import argparse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def chooseModel(data, model):

    config = configparser.ConfigParser()
    config.read('config.config')
    path=config['DEFAULT'][data]
    print('hasdas',type(path))
    if model == 'arima':
        # try:    
        # except:
        #     return 'An exception occured'
        
        data = md.convertIndexToDateDF(path)
        diff = 0
        diff_df = data
        checkstationary = md.checkStationary(data)
        
        while not checkstationary:
            diff_df = md.makeStationary(data)
            checkstationary = md.checkStationary(diff_df)
            diff += 1
        
        p,q = md.getP_Q(diff_df,difference=diff)

        train = data[0:len(data)-10]
        test = data[-10:]

        arima_model = md.modelArima(train,p,diff,q)
        predictions = arima_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

        ar_score = mean_squared_error(test, predictions)
        print(ar_score)
        return ar_score
    
    else:
        data = md.convertIndexToDateDF(path)
        train = data[0:len(data)-10]
        test = data[-10:]
        autoarima = md.model_Autoarima(train)
        pred = autoarima.predict()
        ar_score = mean_squared_error(test, pred)
        print('ARIMA MSE: {}'.format(round(ar_score,4)))
        print(ar_score)
        return ar_score



if __name__ == "__main__": 
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=str)
    parser.add_argument("--model",type=str)
    args=parser.parse_args()
    chooseModel(data=args.data, model=args.model)