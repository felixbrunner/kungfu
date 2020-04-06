import pandas as pd
import numpy as np


class FinancialSeries(pd.Series):

    '''
    A Financial Series is a pandas Series that contains financial observations.
    '''

    _attributes_ = "obstype"

    # constructor properties to output the right types
    @property
    def _constructor(self):
        return FinancialSeries

    @property
    def _constructor_expanddim(self):
        from kungfu.frame import FinancialDataFrame
        return FinancialDataFrame


    def __init__(self, *args, **kwargs):
        super(FinancialSeries, self).__init__(*args, **kwargs)
        self.obstype = None


    def set_obstype(self, obstype=None):
        assert obstype in ['price', 'return', 'logreturn', 'characteristic', None],\
            'obstype needs to be return, price, logreturn, or characteristic'
        self.obstype = obstype


    def convert_prices_to_returns(self):
        '''
        Converts a financial series from observation type price to return.
        '''
        assert self.obstype is 'price', 'obstype is not price'
        returns = self.pct_change()
        returns.obstype = 'return'
        return returns


    def convert_prices_to_logreturns(self):
        '''
        Converts a financial series from observation type price to logreturn.
        '''
        assert self.obstype is 'price', 'obstype is not price'
        logreturns = np.log(self) - np.log(self.shift(1))
        logreturns.obstype = 'logreturn'
        return logreturns


    def convert_returns_to_prices(self, init_price=100): # TODO: adapt for incomplete series
        '''
        Converts a financial series from observation type return to logreturn.
        '''
        assert self.obstype is 'return', 'obstype is not return'
        prices = self+1
        prices.iat[0] = init_price
        prices = prices.cumprod()
        prices.obstype = 'price'
        return prices


    def convert_returns_to_logreturns(self):
        '''
        Converts a financial series from observation type return to logreturn.
        '''
        assert self.obstype is 'return', 'obstype is not return'
        logreturns = np.log(self+1)
        logreturns.obstype = 'logreturn'
        return logreturns


    def convert_logreturns_to_prices(self, init_price=100): # TODO: adapt for incomplete series
        '''
        Converts a financial series from observation type return to logreturn.
        '''
        assert self.obstype is 'logreturn', 'obstype is not return'
        prices = np.exp(self)
        prices.iat[0] = init_price
        prices = prices.cumprod()
        prices.obstype = 'price'
        return prices


    def convert_logreturns_to_returns(self):
        '''
        Converts a financial series from observation type return to logreturn.
        '''
        assert self.obstype is 'logreturn', 'obstype is not return'
        returns = np.exp(self)-1
        returns.obstype = 'return'
        return returns
