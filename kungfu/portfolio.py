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

class Portfolio():

    '''
    Class to hold a collection of assets.
    All inputs should have pandas MultiIndex as index with time index as level 0 and asset index as level 1.
    '''

    def __init__(self, asset_returns=None, weights=None, asset_prices=None, quantities=None):

        assert (asset_returns is not None) != (asset_prices is not None),\
            'need to supply exactly one of asset_returns and asset_prices'

        self.asset_returns = (asset_returns, False)
        self.quantities = (quantities, False)
        self.asset_prices = (asset_prices, False)
        self.weights = (weights, False)
        self._update_merged_data()


    def _prepare_data(self, data):

        '''
        Returns formatted data to be set as portfolio properties.
        '''

        data = FinancialDataFrame(data)

        if type(data.index) == pd.core.indexes.datetimes.DatetimeIndex:
            data = data.stack().to_frame()

        data = data.squeeze().sort_index()

        return data


    @property
    def asset_returns(self):
        return self.__asset_returns

    @asset_returns.setter
    def asset_returns(self, return_data):

        '''
        Sets the contained assets' returns as a FinancialDataFrame.
        '''

        # unpack if a tuple is passed
        try:
             return_data, set_merged = return_data
        except ValueError:
             set_merged = True

        if hasattr(self, 'asset_returns') and self.asset_returns is not None:
            warnings.warn('asset_returns will be overridden')

        if return_data is not None:
            return_data = self._prepare_data(return_data).rename('return')
        self.__asset_returns = return_data

        if set_merged:
            self._update_merged_data()


    def infer_returns(self):

        '''
        Sets asset_returns inferred from asset_prices inplace.
        '''

        assert self.asset_prices is not None,\
            'asset_prices unavailable'

        asset_returns = FinancialSeries(index=self.asset_prices.index)
        for asset in self.assets:
            index = self.asset_prices.index.get_level_values(1)==asset
            asset_returns.loc[index] = FinancialSeries(self.asset_prices.loc[index].squeeze())\
                                            .set_obstype('return')\
                                            .to_returns()
        asset_returns = asset_returns.rename('return')
        pf_inferred = self.__copy__()
        pf_inferred.asset_returns = asset_returns
        return pf_inferred


    @property
    def asset_prices(self):
        return self.__asset_prices


    @asset_prices.setter
    def asset_prices(self, price_data):

        '''
        Returns a FinancialSeries of prices corresponding to the Portfolio's asset_returns.
        '''

        # unpack if a tuple is passed
        try:
             price_data, set_merged = price_data
        except ValueError:
             set_merged = True

        if hasattr(self, 'asset_prices') and self.asset_prices is not None:
            warnings.warn('asset_prices will be overridden')

        if price_data is not None:
            price_data = self._prepare_data(price_data).rename('price')
        self.__asset_prices = price_data

        if set_merged:
            self._update_merged_data()


    def infer_prices(self):

        '''
        Sets asset_prices inferred from asset_returns inplace.
        '''

        assert self.asset_returns is not None,\
            'asset_returns unavailable'

        asset_prices = FinancialSeries(index=self.asset_returns.index)
        for asset in self.assets:
            index = self.asset_returns.index.get_level_values(1)==asset
            asset_prices.loc[index] = FinancialSeries(self.asset_returns.loc[index].squeeze())\
                                            .set_obstype('return')\
                                            .to_prices()
        asset_prices = asset_prices.rename('price')
        pf_inferred  = self.__copy__()
        pf_inferred.asset_prices = asset_prices
        return pf_inferred


    @property
    def quantities(self):
        return self.__quantities

    @quantities.setter
    def quantities(self, quantity_data):

        '''
        Sets returns weighting data as long format FinancialSeries.
        Drops missing observations.
        '''

        # unpack if a tuple is passed
        try:
             quantity_data, set_merged = quantity_data
        except ValueError:
             set_merged = True

        if hasattr(self, 'quantities') and self.quantities is not None:
            warnings.warn('quantities will be overridden')

        if quantity_data is not None:
            quantity_data = self._prepare_data(quantity_data).dropna().rename('quantity')
        self.__quantities = quantity_data

        if set_merged:
            self._update_merged_data()


    def infer_quantities(self, value=1): # EXTEND TO MAKE VALUE DYNAMIC

        '''
        Sets quantities inferred from weights such that the total value of the portfolio is fixed.
        '''

        assert self.asset_prices is not None,\
            'asset_prices unavailable'

        assert self.weights is not None,\
            'weights unavailable'

        merged_data = self.merged_data
        quantities = merged_data['weight'] / merged_data['price'] * value

        pf_inferred = self.__copy__()
        pf_inferred.quantities = quantities.rename('quantity')
        return pf_inferred


    @property
    def weights(self):
        return self.__weights


    @weights.setter
    def weights(self, weight_data):

        '''
        Sets quantity data as long format FinancialSeries.
        Drops missing observations.
        '''

        # unpack if a tuple is passed
        try:
             weight_data, set_merged = weight_data
        except ValueError:
             set_merged = True

        if hasattr(self, 'weights') and self.weights is not None:
            warnings.warn('weights will be overridden')

        if weight_data is not None:
            weight_data = self._prepare_data(weight_data).dropna().rename('weight')
        self.__weights = weight_data

        if set_merged:
            self._update_merged_data()


    def infer_weights(self):

        '''
        Sets weights inferred from quantities inplace.
        '''

        assert self.quantities is not None,\
            'quantities unavailable'

        assert self.asset_prices is not None,\
            'asset_prices unavailable'

        merged_data = self.merged_data
        asset_values = merged_data['price'] * merged_data['quantity']

        total_value = self.value

        weights = asset_values\
                        .to_frame()\
                        .join(total_value**-1, how='left', rsuffix='_div')\
                        .prod(axis=1)

        pf_inferred = self.__copy__()
        pf_inferred.weights = weights.rename('weight')
        return pf_inferred


    def scale_weights(self, total=1):

        '''
        Returns Portfolio object with total weights scaled to 1 in each period.
        '''

        assert self.weights is not None,\
            'weights unavailable'

        total_weights = self.weights\
                                .groupby(self.weights.index.get_level_values(0))\
                                .sum()\
                                .astype(float)
        scaled_weights = self.weights\
                                .to_frame()\
                                .join(total_weights**-1, how='left', rsuffix='_tot')\
                                .prod(axis=1)*total

        pf_scaled = self.__copy__()
        pf_scaled.weights = scaled_weights
        return pf_scaled


    @property
    def value(self):

        '''
        Returns a FinancialSeries that contains portfolio values.
        '''

        assert self.asset_prices is not None,\
            'asset_prices unavailable'

        assert self.quantities is not None,\
            'quantities unavailable'

        merged_data = self.merged_data
        asset_values = merged_data['price'] * merged_data['quantity']
        portfolio_value = asset_values\
                            .groupby(asset_values.index.get_level_values(0))\
                            .sum()
        portfolio_value = FinancialSeries(portfolio_value).rename('value')
        return portfolio_value


    def _merge_data(self):

        '''
        Merges all data saved in the Portfolio object and returns a FinancialDataFrame.
        '''

        if self.asset_returns is not None:
            merged_data = FinancialDataFrame(self.asset_returns)
            if self.asset_prices is not None:
                merged_data = merged_data\
                            .merge(self.asset_prices, how='outer',
                                left_index=True, right_on=self.asset_prices.index.names)
        else:
            merged_data = FinancialDataFrame(self.asset_prices)

        if self.quantities is not None:
            merged_data = merged_data\
                            .merge(self.quantities, how='outer',
                                left_index=True, right_on=self.quantities.index.names)\
                            .sort_index()

        if self.weights is not None:
            merged_data = merged_data\
                            .merge(self.weights, how='outer',
                                left_index=True, right_on=self.weights.index.names)\
                            .sort_index()

        return merged_data


    def _update_merged_data(self):

        '''
        Updates the internally stored combined data.
        '''

        self.__merged_data = self._merge_data()


    @property
    def merged_data(self):
        return self.__merged_data


    @property
    def assets(self):

        '''
        Returns a list of assets in the portfolio.
        '''

        asset_list = list(self.merged_data.index.get_level_values(1).unique())
        return asset_list


    @property
    def start_date(self):

        '''
        Returns the date of the first return observation of the portfolio.
        '''

        return self.asset_returns.unstack().index[0]


    def __copy__(self):
        return copy.deepcopy(self)


    def lag_quantities(self, lags=1):

        '''
        Returns Portfolio object that contains lagged quantities.
        Lags are based on the index of the asset_returns data.
        '''

        lagged_quantities = self.merged_data['quantity'].unstack().shift(lags).stack()

        pf_lagged = self.__copy__()
        pf_lagged.quantities = lagged_quantities
        if pf_lagged.weights is not None:
            pf_lagged = pf_lagged.lag_weights(lags=lags)
        return pf_lagged


    def lag_weights(self, lags=1):

        '''
        Returns Portfolio object that contains lagged weights.
        Lags are based on the index of the asset_returns data.
        '''

        lagged_weigts = self.merged_data['weight'].unstack().shift(lags).stack()

        pf_lagged = self.__copy__()
        pf_lagged.weights = lagged_weigts
        if pf_lagged.quantities is not None:
            pf_lagged = pf_lagged.lag_quantities(lags=lags)
        return pf_lagged


    def set_equal_quantities(self, quantity=1):

        '''
        Sets qunatities such that each asset has a quantity of 1 at the beginning of the sample.
        '''

        pf_equal = self.__copy__()
        pf_equal.quantities = FinancialSeries(quantity, index=pd.MultiIndex.from_product([[self.start_date],self.assets],\
                                                                                        names=self.asset_returns.index.names))
        pf_equal = pf_equal.infer_weights()
        return pf_equal


    def set_equal_weights(self):

        '''
        Sets qunatities such that each asset has equal weight at the beginning of the sample.
        '''

        pf_equal = self.__copy__()
        pf_equal.weights = FinancialSeries(1, index=pd.MultiIndex.from_product([[self.start_date],self.assets],\
                                                                                        names=self.asset_returns.index.names))
        with warnings.catch_warnings(record=True) as w:
            pf_equal = pf_equal.scale_weights()
        pf_equal = pf_equal.infer_quantities()
        return pf_equal


    def _rebalance_continuously(self, **kwargs):

        '''
        Returns quantities with missing quantities filled through continuous rebalancing.
        '''

        pf_rebalanced = self.__copy__()

        if pf_rebalanced.weights is None:
            pf_rebalanced = pf_rebalanced.infer_weights()

        merged_data = pf_rebalanced.merged_data
        filled_weights = merged_data['weight']\
                                .groupby(merged_data.index.get_level_values(1))\
                                .fillna(method='ffill', **kwargs)

        pf_rebalanced.weights = filled_weights
        pf_rebalanced = pf_rebalanced.infer_quantities()
        return pf_rebalanced


    def _rebalance_discretely(self, **kwargs):

        '''
        Returns quantities with missing quantities filled through rebalancing at dates given by the quantity data.
        '''

        pf_rebalanced = self.__copy__()

        if pf_rebalanced.quantities is None:
            pf_rebalanced.quantities = pf_rebalanced.weights.copy()

        merged_data = pf_rebalanced.merged_data
        filled_quantities = merged_data['quantity']\
                                    .groupby(merged_data.index.get_level_values(1))\
                                    .fillna(method='ffill', **kwargs)

        pf_rebalanced.quantities = filled_quantities
        pf_rebalanced = pf_rebalanced.infer_weights()
        return pf_rebalanced


    def rebalance(self, method='discrete'):

        '''
        Rebalances the Portfolio's quantities either discretely (default) or continuously.
        '''

        assert method in ['discrete', 'continuous'],\
            'method must be either discrete or continuous'

        if method == 'continuous':
            pf_rebalanced = self._rebalance_continuously()
        elif method == 'discrete':
            pf_rebalanced = self._rebalance_discretely()

        return pf_rebalanced


    @property
    def returns(self):

        '''
        Returns a FinancialSeries containing portfolio returns.
        '''

        assert self.asset_returns is not None,\
            'asset_returns unavailable'

        assert self.weights is not None,\
            'weights unavailable'

        returns = self.asset_returns * self.weights

        portfolio_returns = returns.\
                                groupby(returns.index.get_level_values(0))\
                                .sum()
        portfolio_returns = FinancialSeries(portfolio_returns)\
                                .set_obstype('return')\
                                .rename('return')
        return portfolio_returns


    @property
    def total_weights(self):

        '''
        Returns a FinancialSeries containing total portfolio weights.
        '''

        total_weights = self.weights\
                                .groupby(self.weights.index.get_level_values(0))\
                                .sum()
        total_weights = FinancialSeries(total_weights)
        return total_weights


    @property
    def delevered_returns(self):

        '''
        Returns a FinancialSeries containing delevered portfolio returns.
        '''

        return self.returns / self.total_weights


    @property
    def turnover(self):

        '''
        Returns a FinancialSeries containing portfolio turnover as a percentage of portfolio value.
        '''

        turnover = self.weights\
                        .groupby(self.weights.index.get_level_values(1))\
                        .diff(1)\
                        .abs()\
                        .groupby(self.weights.index.get_level_values(0))\
                        .sum()
        turnover = turnover / self.total_weights
        return turnover




################################################################################


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

    if method == 'simultaneous':
        portfolio_bins = _bin_simultaneously(sorting_data, n_sorts)
    elif method == 'sequential':
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

    if method == 'percentage':
        assert 0 <= quantity <= 1, 'quantity needs to be between 0 and 1'
        selection = _select_percentage(sorting_data, quantity)

    elif method == 'fixed':
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
