import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from kungfu.frame import FinancialDataFrame
from kungfu.series import FinancialSeries

import kungfu.index as index

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
