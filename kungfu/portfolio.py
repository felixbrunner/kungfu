import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from kungfu.frame import FinancialDataFrame
from kungfu.series import FinancialSeries

import kungfu.index as index

import warnings
import copy

'''
TO DO:
- Output table class
- Output plots
- assert datetime
- long_short_potfolio function
'''


def _generate_portfolio_names(sort_names, n_sorts):

    '''
    Returns a pandas Index or MultiIndex object that contains the names of
    sorted portfolios.
    '''

    assert len(sort_names) == len(list(n_sorts)),\
        'sort_names and n_portfolios length mismatch'

    univariate_indices = []
    for sort,n in zip(sort_names,n_sorts):
        univariate_indices += [[sort+'_low']\
                +[sort+'_'+str(i) for i in range(2,n)]\
                +[sort+'_high']]

    if len(sort_names) == 1:
        return pd.Index(univariate_indices[0])
    else:
        return pd.MultiIndex.from_product(univariate_indices)


def _prepare_return_data(return_data):

    '''
    Returns return data in long format DataFrame.
    '''

    return_data = FinancialDataFrame(return_data)

    if type(return_data.index) == pd.core.indexes.datetimes.DatetimeIndex:
        return_data = return_data.stack().to_frame()

    return_data = return_data.sort_index()
    return return_data


def _prepare_sorting_data(sorting_data):

    '''
    Returns sorting data in long format DataFrame.
    Drops missing observations.
    '''

    if type(sorting_data.index) == pd.core.indexes.datetimes.DatetimeIndex:
        sorting_data = sorting_data.stack()#.to_frame()

    assert type(sorting_data.index) == pd.core.indexes.multi.MultiIndex,\
        'Need to supply panel data as sorting variable'

    sorting_data = FinancialDataFrame(sorting_data).dropna().sort_index()
    return sorting_data


def _bin_simultaneously(sorting_data, n_sorts):

    '''
    Returns a FinancialSeries of portfolio mappings resulting from simultaneous
    sorting.
    '''

    # prepare
    sort_names = list(sorting_data.columns)
    portfolio_name = ', '.join(sort_names)

    # create bins
    for sort, n in zip(sort_names, n_sorts):
        sorting_data[sort] = sorting_data[sort]\
            .groupby(sorting_data.index.get_level_values(1))\
            .apply(lambda x: pd.cut(x.rank(), n, labels=False)+1)

    # output series
    if len(sort_names) == 1:
        portfolio_bins = sorting_data.squeeze()
    else:
        portfolio_bins = sorting_data.apply(tuple, axis=1)

    return FinancialSeries(portfolio_bins, name=portfolio_name)


def _bin_sequentially(sorting_data, n_sorts):

    '''
    Returns a FinancialSeries of portfolio mappings resulting from sequential sorting.
    '''

    # prepare
    sort_names = list(sorting_data.columns)
    portfolio_name = ', '.join(sort_names)

    # create bins
    grouper = [list(sorting_data.index.get_level_values(1))]
    for sort, n in zip(sort_names, n_sorts):
        sorting_data[sort] = sorting_data[sort]\
            .groupby(grouper)\
            .apply(lambda x: pd.cut(x.rank(), n, labels=False)+1)
        grouper += [list(sorting_data[sort].values)]

    # output series
    if len(sort_names) == 1:
        portfolio_bins = sorting_data.squeeze()
    else:
        portfolio_bins = sorting_data.apply(tuple, axis=1)

    return FinancialSeries(portfolio_bins, name=portfolio_name)


def _merge_data_for_portfolios(return_data, portfolio_bins, lag, **kwargs):

    '''
    Returns a joined DataFrame that contains aligned return data and sorting
    data.
    '''

    bins_name = portfolio_bins.name

    # merge
    merged_data = FinancialDataFrame(return_data)\
                        .merge(portfolio_bins, how='outer',
                            left_index=True, right_on=portfolio_bins.index.names)\
                        .sort_index()

    # lag & forward fill
    merged_data[bins_name] = merged_data[bins_name]\
                                .groupby(merged_data.index.get_level_values(0))\
                                .apply(lambda x: x.shift(lag)\
                                            .fillna(method='ffill', **kwargs))

    merged_data = merged_data.dropna()
    return merged_data


def sort_portfolios(return_data, sorting_data, n_sorts=10, lag=1,
                method='simultaneous', **kwargs):

    '''
    Sort returns into portfolios based on one or more sorting variables.
    Method can be simultaneous or sequential.

    TO DO:
    - flexible weights (eg value-weighted)
    '''

    assert method in ['simultaneous', 'sequential'],\
        'method needs to be either simultaneous or sequential'

    return_data = _prepare_return_data(return_data)
    sorting_data = _prepare_sorting_data(sorting_data)

    if type(n_sorts) == int:
        n_sorts = [n_sorts for col in sorting_data.columns]

    if method is 'simultaneous':
        portfolio_bins = _bin_simultaneously(sorting_data, n_sorts)
    elif method is 'sequential':
        portfolio_bins = _bin_sequentially(sorting_data, n_sorts)

    # merge
    merged_data = _merge_data_for_portfolios(return_data,
                                        portfolio_bins, lag, **kwargs)
    portfolio_names = _generate_portfolio_names(sorting_data.columns, n_sorts)

    # create outputs
    results = PortfolioSortResults()
    results.mapping = portfolio_bins
    grouper = [list(merged_data.index.get_level_values(1)),list(merged_data.iloc[:,1])]
    results.size = FinancialSeries(merged_data.iloc[:,1]\
                                    .groupby(grouper).count())
    results.returns = FinancialSeries(merged_data.iloc[:,0]\
                                    .groupby(grouper).mean())
    results.returns.set_obstype('return')
    results.names = portfolio_names

    if results.returns.index.get_level_values(1).dtype == 'float':
        results.size.index = pd.MultiIndex.from_arrays(\
                [results.size.index.get_level_values(0),\
                 results.size.index.get_level_values(1).astype(int)])
        results.returns.index = pd.MultiIndex.from_arrays(\
                [results.returns.index.get_level_values(0),\
                 results.returns.index.get_level_values(1).astype(int)])

    return results


def sort_longshort(return_data, sorting_data, quantity=0.3, lag=1,
                method='percentage', **kwargs):

    '''
    Sort returns into long-short strategy based on one sorting variable.
    Method can be percentage or fixed.

    TO DO:
    - flexible weights (eg value-weighted)
    '''

    assert method in ['percentage', 'fixed'],\
        'method needs to be either percentage or fixed'

    return_data = _prepare_return_data(return_data)
    sorting_data = _prepare_sorting_data(sorting_data)

    if method is 'percentage':
        assert 0 <= quantity <= 1, 'quantity needs to be between 0 and 1'
        selection = _select_percentage(sorting_data, quantity)

    elif method is 'fixed':
        assert quantity <= len(return_data.index.get_level_values(0).unique()),\
            'quantity needs to be less or equal to the number of assets'
        selection = _select_fixed(sorting_data, quantity)

    # merge
    merged_data = _merge_data_for_portfolios(return_data,
                                        selection, lag, **kwargs)

    '''
    # create outputs
    results = PortfolioSortResults()
    results.mapping = portfolio_bins
    grouper = [list(merged_data.index.get_level_values(1)),list(merged_data.iloc[:,1])]
    results.size = FinancialSeries(merged_data.iloc[:,1]\
                                    .groupby(grouper).count())
    results.returns = FinancialSeries(merged_data.iloc[:,0]\
                                    .groupby(grouper).mean())
    results.returns.set_obstype('return')
    results.names = portfolio_names

    if results.returns.index.get_level_values(1).dtype == 'float':
        results.size.index = pd.MultiIndex.from_arrays(\
                [results.size.index.get_level_values(0),\
                 results.size.index.get_level_values(1).astype(int)])
        results.returns.index = pd.MultiIndex.from_arrays(\
                [results.returns.index.get_level_values(0),\
                 results.returns.index.get_level_values(1).astype(int)])'''


    return results



class PortfolioSortResults():

    '''
    Class to hold results of portfolio sorts
    '''

    def __init__(self):
        self.returns = None
        self.size = None
        self.mapping = None
        self.names = None


    def summarise_performance(self, annual_obs=1):

        '''
        Summarises the performance of each sorted portfolio in the
        PortfolioSortResults object.
        '''

        summary = self.returns.unstack().summarise_performance(obstype='return',
                                                    annual_obs=annual_obs)
        return summary


    def plot_indices(self, **kwargs):

        '''
        '''

        fig, ax = plt.subplots(1, 1, **kwargs)

        for (pf_name, pf_returns) in self.returns.unstack().iteritems():
            ax.plot(pf_returns.set_obstype('return').to_prices(init_price=1),
                        linewidth=1, label=pf_name)
        import kungfu.plotting as plotting
        startdate = self.returns.unstack().index[0]
        enddate = self.returns.unstack().index[-1]
        plotting.add_recession_bars(ax, startdate=startdate, enddate=enddate)
        ax.legend(loc='upper left')

        return fig


    @property
    def frequencies(self):
        freq = self.mapping\
            .groupby([self.mapping.index.get_level_values(0), self.mapping.values])\
            .count().unstack(fill_value=0)
        return FinancialDataFrame(freq)


################################################################################


class Portfolio():

    '''
    Class to hold a portfolio of assets.
    '''

    def __init__(self, asset_returns=None, weights=None, asset_prices=None, quantities=None):

        assert (asset_returns is not None) != (asset_prices is not None),\
            'need to supply exactly one of asset_returns and asset_prices'

        self.asset_returns = asset_returns
        self.quantities = quantities
        self.asset_prices = asset_prices
        self.weights = weights


    @property
    def asset_returns(self):
        return self.__asset_returns

    @asset_returns.setter
    def asset_returns(self, return_data):

        '''
        Sets the contained assets' returns as a FinancialDataFrame.
        '''
        if return_data is None:
            return_data = FinancialDataFrame(return_data)

            if type(return_data.index) == pd.core.indexes.datetimes.DatetimeIndex:
                return_data = return_data.stack().to_frame()

            return_data = return_data.squeeze().rename('return').sort_index()

        self.__asset_returns = return_data


    def infer_returns(self):

        '''
        Sets asset_returns inferred from asset_prices inplace.
        '''

        assert self.asset_prices is not None,\
            'asset_prices unavailable'

        if self.asset_returns is not None:
            warnings.warn('asset_returns will be overridden')

        asset_returns = FinancialSeries(index=self.asset_prices.index)
        for asset in self.assets:
            index = self.asset_prices.index.get_level_values(1)==asset
            asset_returns.loc[index] = FinancialSeries(self.asset_prices.loc[index].squeeze())\
                                            .set_obstype('return')\
                                            .to_returns()
        asset_returns = asset_returns.rename('return')

        self.__asset_returns = asset_returns



    @property
    def asset_prices(self):
        return self.__asset_prices


    @asset_prices.setter
    def asset_prices(self, price_data):

        '''
        Returns a FinancialSeries of prices corresponding to the Portfolio's asset_returns.
        '''
        if price_data is not None:
            price_data = FinancialDataFrame(price_data)

            if type(price_data.index) == pd.core.indexes.datetimes.DatetimeIndex:
                price_data = price_data.stack().to_frame()

            price_data = price_data.squeeze().rename('price').sort_index()

        self.__asset_prices = price_data


    def infer_prices(self):

        '''
        Sets asset_prices inferred from asset_returns inplace.
        '''

        assert self.asset_returns is not None,\
            'asset_returns unavailable'

        if self.asset_prices is not None:
            warnings.warn('asset_prices will be overridden')

        asset_prices = FinancialSeries(index=self.asset_returns.index)
        for asset in self.assets:
            index = self.asset_returns.index.get_level_values(1)==asset
            asset_prices.loc[index] = FinancialSeries(self.asset_returns.loc[index].squeeze())\
                                            .set_obstype('return')\
                                            .to_prices()
        asset_prices = asset_prices.rename('price')

        self.__asset_prices = asset_prices


    @property
    def quantities(self):
        return self.__quantities

    @quantities.setter
    def quantities(self, quantity_data):

        '''
        Sets returns weighting data as long format FinancialSeries.
        Drops missing observations.
        '''

        if quantity_data is not None:
            quantity_data = FinancialDataFrame(quantity_data)

            if type(quantity_data.index) == pd.core.indexes.datetimes.DatetimeIndex:
                quantity_data = quantity_data.stack()#.to_frame()

            quantity_data = quantity_data.squeeze().dropna().rename('quantity').sort_index()
        self.__quantities = quantity_data


    def infer_quantities(self): # TO BE IMPLEMENTED
        pass


    @property
    def weights(self):
        return self.__weights


    @weights.setter
    def weights(self, weight_data):

        '''

        '''

        if weight_data is not None:
            weight_data = FinancialDataFrame(weight_data)

            if type(weight_data.index) == pd.core.indexes.datetimes.DatetimeIndex:
                weight_data = weight_data.stack().to_frame()

            weight_data = weight_data.squeeze().rename('weight').sort_index()

        self.__weights = weight_data


    def infer_weights(self):

        '''

        '''

        assert self.quantities is not None,\
            'quantities unavailable'

        weights = self.scale_quantities(1).quantities
        return weights


    @property
    def assets(self):

        '''
        Returns a list of assets in the portfolio.
        '''
        if self.__asset_returns is not None:
            asset_list = list(self.asset_returns.index.get_level_values(1).unique())
        else:
            asset_list = list(self.asset_prices.index.get_level_values(1).unique())

        return asset_list


    @property
    def merged_data(self):

        '''
        Merges the Portfolio's asset_returns data with the weighting_data and returns a FinancialDataFrame.
        '''

        merged_data = FinancialDataFrame(self.asset_returns)

        if self.__asset_prices is not None:
            merged_data = merged_data\
                            .merge(self.asset_prices, how='outer',
                                left_index=True, right_on=self.quantities.index.names)

        if self.quantities is not None:
            merged_data = merged_data\
                            .merge(self.quantities, how='outer',
                                left_index=True, right_on=self.quantities.index.names)\
                            .sort_index()

        return merged_data


    @property
    def start_date(self):

        '''
        Returns the date of the first return observation of the portfolio.
        '''

        return self.asset_returns.unstack().index[0]


    def __copy__(self):
        return copy.deepcopy(self)



    def scale_quantities(self, total=1):

        '''
        Returns FinancialSeries with total weights scaled to 1 in each period.
        '''

        total_quantities = self.quantities\
                                .groupby(self.quantities.index.get_level_values(0))\
                                .sum()\
                                .astype(float)**-1*total
        scaled_quantities = self.quantities\
                                .to_frame()\
                                .join(total_quantities, how='left', rsuffix='_tot')\
                                .prod(axis=1)

        pf_scaled = self.__copy__()
        pf_scaled.quantities = scaled_quantities
        return pf_scaled


    def lag_quantities(self, lags=1):

        '''
        Returns FinancialSeries that contains lagged quantities.
        Lags are based on the index of the asset_returns data.
        '''

        lagged_quantities = self.merged_data['quantity'].unstack().shift(lags).stack()

        pf_lagged = self.__copy__()
        pf_lagged.quantities = lagged_quantities
        return pf_lagged



    def set_equal_quantities(self, quantity=1):

        '''
        Sets qunatities such that each asset has a quantity of 1 at the beginning of the sample.
        '''

        if self.quantities is not None:
            warnings.warn('quantities will be overriden')

        pf_equal = self.__copy__()
        pf_equal.quantities = FinancialSeries(quantity, index=pd.MultiIndex.from_product([[self.start_date],self.assets],\
                                                                                        names=self.asset_returns.index.names))
        return pf_equal


    def _rebalance_continuously(self, **kwargs):

        '''
        Returns quantities with missing quantities filled through continuous rebalancing.
        '''

        merged_data = self.merged_data
        filled_quantities = merged_data['quantity']\
                                .groupby(merged_data.index.get_level_values(1))\
                                .fillna(method='ffill', **kwargs)
        return filled_quantities


    def _rebalance_discretely(self, **kwargs):

        '''
        Returns quantities with missing quantities filled through rebalancing at dates given by the quantity data.
        '''

        merged_data = self.merged_data
        merged_data['quantity'] = merged_data['quantity']\
                                        .groupby(merged_data.index.get_level_values(1))\
                                        .fillna(method='ffill', **kwargs)
        filled_quantities = merged_data.prod(axis=1)
        return filled_quantities


    def rebalance(self, method='discrete'):

        '''
        Rebalances the Portfolio's quantities either discretely (default) or continuously.
        '''

        assert method in ['discrete', 'continuous'],\
            'method must be either discrete or continuous'

        pf_rebalanced = self.__copy__()

        if method is 'continuous':
            pf_rebalanced.quantities = self._rebalance_continuously()
        elif method is 'discrete':
            pf_rebalanced.quantities = self._rebalance_discretely()

        return pf_rebalanced


    @property
    def returns(self):

        '''

        '''

        returns = self.asset_returns * self.quantities

        portfolio_returns = returns.\
                                groupby(merged_data.index.get_level_values(0))\
                                .sum()
        portfolio_returns = FinancialSeries(portfolio_returns)\
                                .set_obstype('return')
        return portfolio_returns


    @property
    def delevered_returns(self):

        '''

        '''

        returns = self.asset_returns * self.scale_quantities(1).quantities

        portfolio_returns = returns.\
                                groupby(merged_data.index.get_level_values(0))\
                                .sum()
        portfolio_returns = FinancialSeries(portfolio_returns)\
                                .set_obstype('return')
        return portfolio_returns


    @property # TO BE IMPLEMENTED
    def turnover(self):
        pass
