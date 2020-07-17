import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle

def demo_linear_regression():
        style.use('ggplot')

        df = quandl.get('WIKI/GOOGL')
        #print((df[['Open', 'High']]))
        #print(df.head())

        df = df[['Adj. Open', 'Adj. High', 
                'Adj. Low', 'Adj. Close', 
                'Adj. Volume']]

        # for the high low percentage change
        df['HL_PCT'] = (df['Adj. High'] - 
                        df['Adj. Close']) / df['Adj. Close'] * 100.0

        # for the percentage change durinng a day
        df['PCT_change'] = (df['Adj. Close'] - 
                        df['Adj. Open']) / df['Adj. Open'] * 100.0

        # forecasting a label
        forecast_col = 'Adj. Close'
        # fillna function is used to fill Nan data with given value. here it is -99999. why -99999 is unknown?
        df.fillna(-99999, inplace=True) # replacing data with -99999. why is it so not clear out

        # math.ceil gives smallest integer not less than x
        forecast_out = int(math.ceil(0.01*len(df)))
        df['label'] = df[forecast_col].shift(-forecast_out)


        X = np.array(df.drop(['label'],1))
        X = preprocessing.scale(X)
        X = X[:-forecast_out]
        X_lately = X[-forecast_out:]
        df.dropna(inplace=True) # to drop out nan values 
        y = np.array(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #clf = LinearRegression()
        #clf = svm.SVR()
        #clf.fit(X_train, y_train)

        #with open('linearregression.pickle', 'wb') as f:
        #        pickle.dump(clf, f)

        pickle_in = open('linearregression.pickle', 'rb') # used tp store classifier value
        clf = pickle.load(pickle_in)
        accuracy = clf.score(X_test, y_test)
        forecast_set = clf.predict(X_lately)
        # print(forecast_set, accuracy, forecast_out)
        df['Forecast'] = np.nan

        last_date = df.iloc[-1].name # to get last date
        last_unix = last_date.timestamp() # to get a timestamp fo last date 
        one_day = 86400
        next_unix = last_unix + one_day

        for i in forecast_set:
                next_date = datetime.datetime.fromtimestamp(next_unix) # to convert date from timestamp
                next_unix += one_day
                df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i] # to add rows of forecast data with pading nan values
        

        df['Adj. Close'].plot()
        df['Forecast'].plot()
        plt.legend(loc=1)
        plt.xlabel('Data')
        plt.ylabel('Price')
        plt.show()

# demo_linear_regression()

# Best fit line, slope and r_squared
def best_fit_line_and_slop_r_squared():

        from statistics import mean

        style.use('fivethirtyeight')

        x = np.array([1,2,3,4,5], dtype=np.float64)
        y = np.array([4,5,6,7,9], dtype=np.float64)

        def best_fit_slope_and_intercept(x ,y):
                slope = ( (mean(x)*mean(y) - mean(x*y)) /
                        (mean(x)**2 - mean(x**2)) )

                intercept = mean(y) - slope*mean(x)
                return slope, intercept 

        m, b = best_fit_slope_and_intercept(x, y)

        def squared_error(y_orig, y_line):
                return sum((y_line-y_orig)**2)

        def coefficent_of_deterrmination(y_orig, y_line):
                y_mean_line = [mean(y_orig) for y in y_orig]                
                squared_error_regr = squared_error(y_orig, y_line) 
                squared_error_y_mean = squared_error(y_orig, y_mean_line)
                return 1 - (squared_error_regr / squared_error_y_mean)

        regression_line = [ (m*i)+b for i in x]

        r_squared = coefficent_of_deterrmination(y,regression_line)
        print(r_squared)

        predict_x = 8
        predict_y = (m*predict_x) + b

        plt.scatter(x, y)
        plt.scatter(predict_x, predict_y)
        plt.plot(x, regression_line)
        plt.show()

best_fit_line_and_slop_r_squared()

