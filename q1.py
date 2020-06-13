import numpy as np
import pandas as pd
from numpy.random import RandomState
from collections import Counter
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        #self.X_validation = X_validation
        #self.Y_validation = Y_validation
        
    def fit_test(self, X_test):
        self.X_test = X_test
        
    def euclidean_distance(self,row):
        return np.sqrt(np.sum((self.X_train - row) ** 2, axis=1))

    def accuracy(self,y_real, y_pred):
#    print("y_real ",len(y_real))
#    print("y_pred ",len(y_pred))
        accuracy = np.sum(y_real == y_pred) / len(y_real)
        return accuracy

    def manhattan_distance(self,row):
        return np.sum(np.abs(self.X_train-row), axis = 1)
    
    def prediction(self,row,k):
        dist = self.euclidean_distance(row)
        indexes = np.argsort(dist)[:k]
        neighbors = self.Y_train[indexes]
        match = Counter(neighbors).most_common(1)
        #print(match[0][0])
        return match[0][0]
    
    def prediction_manhattan(self,row,k):
        dist = self.manhattan_distance(row)
        indexes = np.argsort(dist)[:k]
        neighbors = self.Y_train[indexes]
        match = Counter(neighbors).most_common(1)
        #print(match[0][0])
        return match[0][0]
    
    def predict_knn(self,k):
        y_pred = [self.prediction(x,k) for x in self.X_test]
        return np.array(y_pred)
    
    def predict_euclidean(self,k):
        y_pred = [self.prediction(x,k) for x in self.X_test]
        return np.array(y_pred)
    
    def predict_manhattan(self,k):
        y_pred = [self.prediction_manhattan(x,k) for x in self.X_test]
        return np.array(y_pred)
    
    def train(self,path):
        df = pd.read_csv(str(path),header = None)
        #rng = RandomState()
        #train = df.sample(frac=0.8,random_state = rng)
        #validation = df.loc[~df.index.isin(train.index)]
        X_train,Y_train = df.iloc[:,1:], df.iloc[:,0]
        #X_validation,Y_validation = validation.iloc[:, 1:], validation.iloc[:,0]
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        #X_validation = np.array(X_validation)
        #Y_validation = np.array(Y_validation)
        self.fit(X_train,Y_train)
        
    def predict(self, path):
        df_test = pd.read_csv(str(path),header = None)
        #print(df_test.shape)
        X_test = df_test.to_numpy()
        self.fit_test(X_test)
        #print(len(X_test))
        # Y_temp_test = list()
        # with open("/media/indranil/New Volume/second sem/SMAI/Assignment 1/q1/dataset/test_labels.csv") as f:
        # for line in f:
        #     if(line == '\n'):
        #         continue
        #     Y_temp_test.append(int(line))
        # Y_test = np.array(Y_temp_test)
        predictions_k = self.predict_euclidean(3)
        return predictions_k
        #accuracy(Y_test, predictions_k)