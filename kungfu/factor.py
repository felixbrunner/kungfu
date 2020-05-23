import scipy as sp
import pandas as pd
import numpy as np

import statsmodels.api as sm
import matplotlib.pyplot as plt

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

    if isinstance(factor_data, pd.Series):
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
        Fit the factor model to a set of assets and return the results.
        '''

        asset_data = _prepare_asset_data(asset_data)
        asset_names = list(asset_data.columns)

        data = _combine_data(self.factor_data, asset_data)

        results = FactorModelResults(self.factor_names, asset_names,
                                        data.index)
        results.asset_means = asset_data.mean()
        results.factor_means = self.factor_data.mean()

        for asset in asset_names:
            estimate = sm.OLS(data[asset],
                            sm.add_constant(data[self.factor_names]),
                            missing='drop')\
                        .fit()
            results.alphas.at[asset] = estimate.params['const']
            results.alphas_se.at[asset] = estimate.bse['const']
            results.betas.loc[asset,self.factor_names] = \
                                    estimate.params[self.factor_names].values
            results.betas_se.loc[asset,self.factor_names] = \
                                    estimate.bse[self.factor_names].values
            results.residuals.loc[:,asset] = estimate.resid.values
            results.idiosyncratic_volas.at[asset] = estimate.mse_resid**0.5
            results.r_squares.at[asset] = estimate.rsquared

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
        self.alphas = FinancialSeries(dtype='float', index=asset_names, name='alpha')
        self.alphas_se = FinancialSeries(dtype='float', index=asset_names, name='alpha_se')
        self.betas = FinancialDataFrame(dtype='float', index=asset_names, columns=self.factor_names)
        self.betas_se = FinancialDataFrame(dtype='float', index=asset_names, columns=self.factor_names)
        self.residuals = FinancialDataFrame(dtype='float', index=timeline, columns=asset_names)
        self.idiosyncratic_volas = FinancialSeries(dtype='float', index=asset_names,
                                                name='idiosyncratic_volatility')
        self.r_squares = FinancialSeries(dtype='float', index=asset_names, name='r_squared')
        self.asset_means = None
        self.factor_means = None


    @property
    def estimates(self):

        '''
        Returns parameter estimates of FactorModelResults as a DataFrame.
        '''

        estimates = self.alphas.to_frame()\
                        .merge(self.betas, how='outer',
                            left_index=True, right_index=True)
        return estimates


    @property
    def expected_returns(self, annual_obs=1):

        '''
        Returns expected returns from the factor model estimates.
        '''

        expected_returns = self.betas\
                            .multiply(self.factor_means).sum(axis=1)*annual_obs
        return expected_returns


    def plot_predictions(self, annual_obs=1, **kwargs):

        '''
        Plots the factor model's predictions against the realisations in the
        sample together with the 45-degree line.
        '''

        expected_returns = self.calculate_expected_returns(annual_obs)

        fig, ax = plt.subplots(1, 1, **kwargs)

        ax.scatter(expected_returns, self.asset_means*annual_obs,
                    label='Test assets',
                    marker='x')
        limits = (max(ax.get_xlim()[0],ax.get_ylim()[0]),\
                  min(ax.get_xlim()[1],ax.get_ylim()[1]))
        ax.plot(limits, limits,
                    clip_on=True, scalex=False, scaley=False,
                    label='45° Line',
                    c='k', linewidth=1, linestyle=':')
        ax.set_xlabel('Expected Return')
        ax.set_ylabel('Realised Return')
        ax.legend(loc='lower right')

        return fig


    def plot_results(self, annual_obs=1, **kwargs):

        '''
        Plots the factor model's estimates in 4 subplots:
        - alphas
        - betas
        - mean returns
        - r squares
        '''

        fig, axes = plt.subplots(4, 1, **kwargs)

        axes[0].errorbar(range(1,len(self.alphas)+1), self.alphas*annual_obs,
                                    yerr=self.alphas_se*annual_obs, fmt='-o')
        axes[0].axhline(0, color='grey', linestyle='--', linewidth=1)
        axes[0].set_title('Annual alphas & standard errors')
        axes[0].set_xticks(range(1,len(self.alphas)+1))
        axes[0].set_xticklabels([])
        #axes[0].xaxis.set_tick_params(labeltop=True, labelbottom=False)
        #axes[0].set_xticklabels(self.alphas.index, rotation='vertical', y=1.1)

        for (factor_name, beta_data) in self.betas.iteritems():
            axes[1].errorbar(range(1,len(self.betas)+1), beta_data,
                                    yerr=self.betas_se[factor_name], fmt='-o')
        axes[1].axhline(0, color='grey', linestyle='--', linewidth=1)
        axes[1].axhline(1, color='grey', linestyle=':', linewidth=1)
        axes[1].set_title('Factor loadings (betas) & standard errors')
        axes[1].set_xticks(range(1,len(self.alphas)+1))
        axes[1].legend(loc='upper left')
        axes[1].set_xticklabels([])

        axes[2].plot(self.asset_means*annual_obs, marker='o')
        axes[2].set_title('Mean Return')
        axes[2].set_xticks(range(1,len(self.alphas)+1))
        axes[2].set_xticklabels([])

        axes[3].plot(self.r_squares, marker='o')
        axes[3].set_title('R²')
        axes[3].set_xticks(range(1,len(self.alphas)+1))
        axes[3].set_xticklabels(self.r_squares.index)#, rotation='vertical')

        return fig
