import scipy as sp
import pandas as pd
import numpy as np

import statsmodels.api as sm

import warnings

from kungfu.frame import FinancialDataFrame




class FactorModel():

    '''
    FactorModel class
    '''

    def __init__(data, factors, assets=None):
        assert all factors in fdf.new_columns, 'all factors need to be in data'
        assert all assets in fdf.new_columns, 'all assets need to be in data'

        self.data = fdf
        self.factors = factors
        self.assets = assets


    def add_assets(self, asset_data):
        self.assets = data


    def fit(self):

        '''
        '''

        assert self.assets is not None, 'add assets to fit model'

        fdf_estimates = kf.FinancialDataFrame(index=self.assets)
        for asset in assets.columns[]:

            estimate = sm.OLS(asset, sm.add_constant(self.factors), missing=missing)\
                        .fit()
