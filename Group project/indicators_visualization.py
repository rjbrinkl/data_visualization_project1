import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from scipy.stats import entropy
from matplotlib import pyplot as plt
# from sklearn import metrics
# import math
# import numpy

def extractData():

    #read in CGMData and InsulinData
    adultData = pd.read_csv('./Cleaned_Data.csv', sep = ',', dtype = 'unicode')
    print(adultData)
    
    

if __name__ == '__main__':
    extractData()