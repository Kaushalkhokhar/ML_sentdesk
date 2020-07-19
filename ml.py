import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm, neighbors
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
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]

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
        plt.legend(loc=2)
        plt.xlabel('Data')
        plt.ylabel('Price')
        plt.show()

#demo_linear_regression()

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

#best_fit_line_and_slop_r_squared()

# testing assumption
def test_assumptioin():

        from statistics import mean
        import random

        style.use('fivethirtyeight')       

        def create_dataset(hm, variance, step=2, correlation=False):
                val = 1
                ys = []
                for i in range(hm):
                        y = val + random.randrange(-variance, variance)
                        ys.append(y)
                        if correlation and correlation == 'pos':
                                val += step
                        elif correlation and correlation == 'neg':
                                val -= step

                xs = [ i for i in range(hm)]

                return np.array(ys, dtype=np.float64), np.array(xs, dtype=np.float64) 
        
        y, x = create_dataset(40, 40, 2, correlation='pos')

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
        #plt.scatter(predict_x, predict_y)
        plt.plot(x, regression_line)
        plt.show()

#test_assumptioin()

# Classification

def KNeighbors_Classifier():
        df = pd.read_csv("breast-cancer-wisconsin.data")
        df.replace('?', -99999, inplace=True)
        df.drop(['id'], 1, inplace=True)

        X = np.array(df.drop(['class'], 1))
        y = np.array(df['class'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        print(accuracy)

        example_measure = np.array([[4,2,1,1,1,2,3,2,1]]) # double bracket is for len at bottom line
        example_measure = example_measure.reshape(len(example_measure),-1)

        example_predict = clf.predict(example_measure)
        print(example_predict)

# KNeighbors_Classifier()

# Euclidean Distance

def KNeighbors_alogorithm():
        from math import sqrt
        import warnings
        from collections import Counter

        style.use('fivethirtyeight')

        dataset = {'k': [[1,2], [2,3], [3,1]], 
                        'r': [[6,5], [5,7], [8,6]]}
        new_features = [5,7]

        def k_nearest_neighbors(data, predict, k=3):
                if len(data) >= k:
                        warnings.warn('K is set ot a value less than total voting group')
                distances = []
                for group in data:
                        for features in data[group]:
                                euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict)) # this is same as squrt( (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 )
                                # or
                                # euclidean_distance = np.array(np.sum(np.array(features)-np.array(predict))**2) # this is same as squrt( (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 )
                                distances.append([euclidean_distance, group])
                
                votes = [i[1] for i in sorted(distances)[:k]]
                print(votes)
                print(Counter(votes).most_common(1))
                vote_result = Counter(votes).most_common(1)[0][0]

                return vote_result
        
        result = k_nearest_neighbors(dataset, new_features, k=3)
        print(result)

        # for i in dataset:
        #         print(i)
        #         for ii in dataset[i]:
        #                 print(ii)
        #                 plt.scatter(ii[0], ii[1], s=100, color=i)
        # or use below
        # [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in i] for i in dataset]
        
        #plt.show()

#KNeighbors_alogorithm()

# Applying KNN to data

def applying_knn_to_data():
        from math import sqrt
        import warnings
        from collections import Counter
        import random               

        def k_nearest_neighbors(data, predict, k=3):
                if len(data) >= k:
                        warnings.warn('K is set ot a value less than total voting group')
                distances = []
                for group in data:
                        for features in data[group]:
                                euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict)) # this is same as squrt( (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 )
                                distances.append([euclidean_distance, group])
                
                votes = [i[1] for i in sorted(distances)[:k]]                
                vote_result = Counter(votes).most_common(1)[0][0]
                confidence =  Counter(votes).most_common(1)[0][1] / k # Infavour votes out of k

                return vote_result, confidence
        
        df = pd.read_csv("breast-cancer-wisconsin.data")
        df.drop(['id'], 1, inplace=True)
        df.replace('?', -99999, inplace=True)
        full_data = df.astype(float).values.tolist()
        random.shuffle(full_data)

        train_size = 0.2
        train_set = {2:[], 4:[]}
        test_set = {2:[], 4:[]}
        train_data = full_data[:-int(len(full_data)*train_size)] 
        test_data = full_data[-int(len(full_data)*train_size):]

        for i in train_data:
                train_set[i[-1]].append(i[:-1])

        for i in test_data:
                test_set[i[-1]].append(i[:-1])

        correct = 0
        total = 0

        for group in test_set:
                for data in test_set[group]:
                        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
                        if group == vote:
                                correct += 1
                        total += 1

        print('accuracy:', correct/total ) 

# applying_knn_to_data()

class Support_Vector_Machine():

        def __init__(self, visulization=True):
                self.visulization = visulization
                self.color = {-1:'r', 1: 'b'}
                if self.visulization:
                        self.fig = plt.figure()
                        self.ax = self.fig.subplot(1,1,1)

        def fit(self, data):
                self.data = data
                # || W ||
                opt_dict = {}

                transform = [ [1, 1],
                              [-1, 1],
                              [-1, -1],
                              [1, -1] ]

                all_data = []
                for yi in seef.data:
                        for fetures in self.data[yi]:
                                for feture in fetures:
                                        all_data.append(fetures)

                self.max_feature_value = max(all_data)
                self.min_feature_value = min(all_data)
                all_data = None

                step_sizes = [self.max_feature_value*0.1,
                              self.max_feature_value*0.01,
                              # point of expense
                              self.max_feature_value*0.001]

                # extremely expensive
                b_range_multiple = 5
                #
                b_multiple = 5

                latest_optimum = self.max_feature_value*10

                for step in step_sizes:
                        w = np.array([latest_optimum, latest_optimum])
                        # we can do this because convex
                        optimized = False
                        while not optimized:
                                pass
        
        def predict(self, features):
                # sign(x.w + b)
                classification = np.sign(np.dot(np.aaray(features), self.w) + self.b)

                return classification

        
data = {-1:np.array([[1, 5],
                     [2, 7],
                     [1, 6],]), 
          1:np.array([[5, 2],
                      [6, -1],
                      [7, 3],])}
