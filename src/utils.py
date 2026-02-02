import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
def haversine_distance(lat1:float, lon1:float, lat2:float, lon2:float):
    """
    Calculate haversine distances between two points given their latitude and
    longitude coordinates
    """
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(
        dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def euclidean_distance(lat1:float, lon1:float, lat2:float, lon2:float) -> np.float32:
    """
    Calculates euclidean distances between two points given their latitude and
    longitude coordinates
    """
    
    location_1 = (lat1,lon1)
    location_2 = (lat2,lon2)
    euclidean = np.sqrt(np.square(location_1[0] - location_2[0]) + np.square(location_1[1] - location_2[1]))
    
    return euclidean


def manhattan_distance(lat1:float, lon1:float, lat2:float, lon2:float) -> np.float32:
    """
    Calculates manhattan distances between two points given their latitude and
    longitude coordinates
    """
    
    location_1 = (lat1,lon1)
    location_2 = (lat2,lon2)
    manhattan = np.abs(location_1[0] - location_2[0]) + np.abs(location_1[1] - location_2[1])
    
    return manhattan


class OutliersRemover(TransformerMixin,OneToOneFeatureMixin, BaseEstimator):
    
    def __init__(self, percentile_values:list,col_subset:list):
        self.percentile_values = percentile_values
        self.col_subset = col_subset
        
    def fit(self,X,y=None):
        # make a copy of X
        X = X.copy()
    
        self.quantiles_ = []
        
        for col in self.col_subset:
            lower_bound = X.loc[:,col].quantile(q=self.percentile_values[0])
            upper_bound = X.loc[:,col].quantile(q=self.percentile_values[1])
            
            self.quantiles_.append((lower_bound,upper_bound))
                  
        return self 
        
    def transform(self,X):
        X = X.copy()
       
        for ind,col in enumerate(self.col_subset):
            lower_bound, upper_bound = self.quantiles_[ind]
            filter_df = X[(X.loc[:,col] >= lower_bound) & (X.loc[:,col] <= upper_bound)]
            X = filter_df
            
        return X