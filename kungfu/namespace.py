import numpy as np
import pandas as pd


@pd.api.extensions.register_series_accessor("kf")
class KungfuAccessor:
    def __init__(self, s):
        self._series = s


    def returns_to_prices(self, init_price=1):

        '''
        Returns a Series that has been converted from returns to prices.
        '''

        returns = self._series
        prices = returns+1
        start_index = returns.kf.find_first_observation(output='row')-1
        prices.iat[start_index] = init_price
        prices = prices.cumprod()

        return prices


    def find_first_observation(self, output='full'):

        '''
        Finds the first available observation and returns it with its index.
        Inputs:
        output - full, index, row, or value
        '''

        assert output in ['full', 'index', 'row', 'value'],\
            'output needs to be full, index, row, or value'

        series = self._series
        observation_count = series.notna().cumsum()
        first_observation = series[observation_count==1]

        if output is 'full':
            return first_observation

        elif output is 'index':
            return first_observation.index[0]

        elif output is 'row':
            return int(np.arange(0,len(series))[observation_count==1])

        elif output is 'value':
            return first_observation.values[0]



@pd.api.extensions.register_dataframe_accessor("kf")
class KungfuAccessor:
    def __init__(self, df):
        self._frame = df
