import scipy as sp
import pandas as pd
import numpy as np

import statsmodels.api as sm

import warnings

from kungfu.frame import FinancialDataFrame
from kungfu.series import FinancialSeries

'''
TO DO:
- Ensure data is of obstype return
'''


def _prepare_asset_data(asset_data):

    '''
    Returns asset timeseries data as a DataFrame in wide format.
    '''

    if type(asset_data.index) == pd.core.indexes.multi.MultiIndex:
        assert len(asset_data.columns) == 1,\
            'too many columns, supply only return data'
        asset_data = asset_data.unstack()

    assert type(asset_data.index) == pd.core.indexes.datetimes.DatetimeIndex,\
        'time index of supplied data needs to be pandas.DatetimeIndex'

    if type(asset_data) == pd.core.series.Series:
        asset_data = asset_data.to_frame()

    return asset_data


def _prepare_factor_data(factor_data):

    '''
    Returns factor timeseries data as a DataFrame in wide format.
    '''

    assert type(factor_data.index) == pd.core.indexes.datetimes.DatetimeIndex,\
        'index of supplied data needs to be pandas.DatetimeIndex'

    if type(factor_data) == pd.core.series.Series:
        factor_data = factor_data.to_frame()

    return factor_data


def _combine_data(factor_data, asset_data):

    '''
    Returns a joined DataFrame that contains aligned asset data and factor data.
    '''

    factor_data = _prepare_factor_data(factor_data)
    asset_data = _prepare_asset_data(asset_data)

    combined_data = asset_data.merge(factor_data, how='left',
                    left_index=True, right_index=True)

    return combined_data


class FactorModel():

    '''
    FactorModel class for the estimation and testing of linear factor models as
    in the asset pricing academic literature.
    '''

    def __init__(self, factor_data):
        self.factor_data = _prepare_factor_data(factor_data)
        self.factor_names = list(self.factor_data.columns)


    def fit(self, asset_data):

        '''
        Fit method
        '''

        asset_data = _prepare_asset_data(asset_data)
        asset_names = list(asset_data.columns)

        data = _combine_data(self.factor_data, asset_data)

        results = FactorModelResults(self.factor_names, asset_names,
                                        data.index)

        for asset in asset_names:
            estimate = sm.OLS(data[asset],
                            sm.add_constant(data[self.factor_names]),
                            missing='drop')\
                        .fit()
            results.alphas.at[asset] = estimate.params['const']
            results.betas.loc[asset,self.factor_names] = estimate.params[self.factor_names].values
            results.residuals.loc[:,asset] = estimate.resid.values

        return results


    def calculate_grs_test(self, asset_data):

        '''
        Returns the GRS test statistic and its corresponding p-Value for
        testing a cross-sectional asset-pricing model as in
        Gibbons/Ross/Shanken (1989).

        Hypothesis: alpha1 = alpha2 = ... = alphaN = 0
        That is if the alphas from N time series regressions on N test assets
        are jointly zero.

        Based on Cochrane (2001) Chapter 12.1
        '''

        asset_data = _prepare_asset_data(asset_data)
        #asset_names = list(asset_data.columns)

        # dimensions
        T = len(asset_data)
        N = len(asset_data.columns)
        K = len(self.factor_names)

        # factor timeseries means and VCV
        factor_means = np.matrix(self.factor_data.mean()).T
        factor_vcv = np.matrix(self.factor_data.cov())

        # timeseries regressions
        ts_regressions = self.fit(asset_data)
        alphas = np.matrix(ts_regressions.alphas).T
        #betas = ts_regressions.betas
        residuals = ts_regressions.residuals

        # asset VCV
        asset_vcv = (T-1)/(T-1-K)*np.matrix(residuals.cov())

        # GRS F-statistic
        f_statistic = (T-N-K)/N \
                *(1+factor_means.T*np.linalg.pinv(factor_vcv)*factor_means)**-1\
                *alphas.T*np.linalg.pinv(asset_vcv)*alphas

        # p-Value for GRS statistic: GRS ~ F(N,T-N-K)
        p_value = 1-sp.stats.f.cdf(f_statistic, N, T-N-K)

        return f_statistic.item(), p_value.item()


class FactorModelResults():

    '''
    Class to hold factor model results:
    - alphas
    - betas
    - pricing errors (residuals)
    '''

    def __init__(self, factor_names, asset_names, timeline):
        self.factor_names = factor_names
        self.asset_names = asset_names
        self.timeline = timeline
        self.alphas = FinancialSeries(index=asset_names, name='alpha')
        self.betas = FinancialDataFrame(index=asset_names, columns=self.factor_names)
        self.residuals = FinancialDataFrame(index=timeline, columns=asset_names)


    def get_estimates(self):

        '''
        Returns parameter estimates of FactorModelResults as a DataFrame.
        '''

        estimates = self.alphas.to_frame().merge(self.betas, how='outer',
                        left_index=True, right_index=True)
        return estimates
