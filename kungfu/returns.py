import pandas as pd
import numpy as np


class FinancialSeries(pd.Series):

    '''
    A Financial Series is a pandas Series that contains financial observations.
    '''

    def __init__(self, obstype='returns', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstype = obstype


    @property
    def _constructor(self):
        return FinancialSeries


    def convert_prices_to_returns(self, return=False):

        '''
        Converts a financial series from observation type prices to returns.
        '''

        assert self.obstype is 'prices', 'observation type is not prices'
        prices = self
        returns = prices/prices.lag(1)-1
        self.values = returns.values
        self.obstype = 'returns'
        if return:
            return self


    def convert_prices_to_logreturns(self, return=False):

        '''
        Converts a financial series from observation type prices to logreturns.
        '''

        assert self.obstype is 'prices', 'observation type is not prices'
        prices = self
        returns = np.log(prices) - np.log(prices.lag(1))
        self.values = returns.values
        self.obstype = 'logreturns'
        if return:
            return self
