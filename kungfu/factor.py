import scipy as sp
import pandas as pd
import numpy as np

import statsmodels.api as sm

import warnings

from kungfu.frame import FinancialDataFrame, FinancialSeries


class FactorModel():

    '''
    FactorModel class
    '''

    def __init__(self, factor_data):
        assert type(factor_data.index) == pd.core.indexes.datetimes.DatetimeIndex,\
            'index of supplied data needs to be pandas.DatetimeIndex'

        self.factor_data = factor_data
        if type(factor_data) == pd.core.series.Series:
            factor_data = factor_data.to_frame()

        self.factor_names = list(factor_data.columns)


    def fit(self, asset_data):

        '''
        Fit method
        '''

        if type(asset_data.index) == pd.core.indexes.multi.MultiIndex:
            assert len(asset_data.columns) == 1,\
                'too many columns, supply only return data'
            asset_data = asset_data.unstack()

        assert type(asset_data.index) == pd.core.indexes.datetimes.DatetimeIndex,\
            'time index of supplied data needs to be pandas.DatetimeIndex'

        if type(asset_data) == pd.core.series.Series:
            asset_data = asset_data.to_frame()

        asset_names = list(asset_data.columns)

        asset_data = asset_data.merge(self.factor_data, how='left',
                        left_index=True, right_index=True)


        results = FactorModelResults(self.factor_names, asset_names)

        for asset in asset_names:
            estimate = sm.OLS(asset_data[asset],
                            sm.add_constant(asset_data[self.factor_names]),
                            missing='drop')\
                        .fit()
            results.alphas.at[asset] = estimate.params['const']
            results.betas.loc[asset,self.factor_names] = estimate.params[self.factor_names].values

        return results



class FactorModelResults():

    '''
    Class to hold factor model results:
    - alphas
    - betas
    '''

    def __init__(self, factor_names, asset_names):
        self.factor_names = factor_names
        self.asset_names = asset_names
        self.alphas = FinancialSeries(index=asset_names, name='alpha')
        self.betas =FinancialDataFrame(index=asset_names, columns=self.factor_names)


    def get_estimates(self):

        '''
        Returns parameter estimates of FactorModelResults
        '''

        estimates = self.alphas.to_frame().merge(self.betas, how='outer',
                        left_index=True, right_index=True)
        return estimates
