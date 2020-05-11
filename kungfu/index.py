import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from kungfu.frame import FinancialDataFrame
from kungfu.series import FinancialSeries

'''
'''


def _prepare_return_data(return_data):

    '''
    Returns return data in long format DataFrame.
    '''

    return_data = FinancialDataFrame(return_data)

    if type(return_data.index) == pd.core.indexes.datetimes.DatetimeIndex:
        return_data = return_data.stack().to_frame()

    return_data = return_data.sort_index()
    return return_data


def _prepare_weighting_data(weighting_data):

    '''
    Returns weighting data in long format DataFrame.
    Drops missing observations.
    '''

    if type(weighting_data.index) == pd.core.indexes.datetimes.DatetimeIndex:
        weighting_data = weighting_data.stack()#.to_frame()

    assert type(weighting_data.index) == pd.core.indexes.multi.MultiIndex,\
        'Need to supply panel data as sorting variable'

    weighting_data = FinancialSeries(weighting_data).dropna().sort_index()
    return weighting_data


def _merge_data_for_index(return_data, weighting_data, lag, **kwargs):

    '''
    Returns a joined DataFrame that contains aligned return data and weighting
    data.
    '''

    weights_name = weighting_data.name

    # merge
    merged_data = FinancialDataFrame(return_data)\
                        .merge(weighting_data, how='outer',
                            left_index=True, right_on=weighting_data.index.names)\
                        .sort_index()

    # lag & forward fill & scale
    merged_data[weights_name] = merged_data[weights_name]\
                                .groupby(merged_data.index.get_level_values(0))\
                                .apply(lambda x: x.shift(lag)\
                                            .fillna(method='ffill', **kwargs))\
                                .groupby(merged_data.index.get_level_values(1))\
                                .apply(lambda x: x.divide(x.sum()))
    merged_data = merged_data.dropna()

    return merged_data


def create_index(return_data, weighting_data=None, lag=0, **kwargs):

    '''
    Returns a FinancialSeries that contains returns of an equal or weighted
    index.
    Weights sum up to one in each period.
    '''

    return_data = _prepare_return_data(return_data)

    # case without variable weights
    if weighting_data is None:
        index_returns = return_data\
                            .groupby(return_data.index.get_level_values(1))\
                            .mean()
        index_returns = FinancialDataFrame(index_returns)\
                            .squeeze()\
                            .rename('equal_index')\
                            .sort_index()

    # case with variable weights
    else:
        # prepare & merge
        weighting_data = _prepare_weighting_data(weighting_data)
        merged_data = _merge_data_for_index(return_data, weighting_data,
                                                lag, **kwargs)

        # returns = sum(weights*returns)
        index_returns = merged_data\
                            .prod(axis=1)\
                            .groupby(merged_data.index.get_level_values(1))\
                            .sum()
        index_returns = FinancialDataFrame(index_returns)\
                            .squeeze()\
                            .rename('weighted_index')

    index_returns = index_returns.set_obstype('return')

    return index_returns
