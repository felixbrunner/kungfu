import scipy as sp
import pandas as pd
import numpy as np
import warnings


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
        assert obstype in ['price',
                           'return',
                           'logreturn',
                           'characteristic',
                           None],\
            'obstype needs to be return, price, logreturn, or characteristic'
        self.obstype = obstype


    def to_returns(self):
        '''
        Converts a financial series to obstype return.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype is not price, return, or logreturn'

        if self.obstype is 'return':
            return self

        elif self.obstype is 'price':
            returns = self.pct_change()
            returns.obstype = 'return'
            return returns

        elif self.obstype is 'logreturn':
            returns = np.exp(self)-1
            returns.obstype = 'return'
            return returns


    def to_logreturns(self):
        '''
        Converts a financial series to obstype logreturn.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype is not price, return, or logreturn'

        if self.obstype is 'logreturn':
            return self

        elif self.obstype is 'price':
            logreturns = np.log(self) - np.log(self.shift(1))
            logreturns.obstype = 'logreturn'
            return logreturns

        elif self.obstype is 'return':
            logreturns = np.log(self+1)
            logreturns.obstype = 'logreturn'
            return logreturns


    def to_prices(self, init_price=1):
        '''
        Converts a financial series to obstype prices.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype is not price, return, or logreturn'

        if self.obstype is 'price':
            if init_price is None:
                return self
            else:
                warnings.warn('rescaling prices, previous values will be lost')
                first_price = self.find_first_observation(output='value')
                prices = self / first_price * init_price
                prices.obstype = 'price'
                return prices

        elif self.obstype is 'return':
            prices = self+1
            start_index = self.find_first_observation(output='row')-1
            prices.iat[start_index] = init_price
            prices = prices.cumprod()
            prices.obstype = 'price'
            return prices

        elif self.obstype is 'logreturn':
            prices = np.exp(self)
            start_index = self.find_first_observation(output='row')-1
            prices.iat[start_index] = init_price
            prices = prices.cumprod()
            prices.obstype = 'price'
            return prices


    def to_obstype(self, type, *args, **kwargs):
        '''
        Converts a financial series to input obstype.
        '''
        assert type in ['price','return','logreturn'],\
            'type needs to be price, return, or logreturn'
        if type is 'price':
            return self.to_prices(*args, **kwargs)
        elif type is 'return':
            return self.to_returns()
        elif type is 'logreturn':
            return self.to_logreturns()


    def to_index(self):
        '''
        Converts a financial series into an index starting from 100.
        '''
        return self.to_prices(init_price=100)


    def to_cumreturns(self):
        '''
        Converts a financial series to cumulative returns.
        '''
        return self.to_prices(init_price=1)-1


    def find_first_observation(self, output='full'):
        '''
        Finds the first available observation and returns it with its index.
        '''
        assert output in ['full', 'index', 'row', 'value'],\
            'output needs to be full, index, row, or value'

        observation_count = self.notna().cumsum()
        first_observation = self[observation_count==1]

        if output is 'full':
            return first_observation

        elif output is 'index':
            return first_observation.index[0]

        elif output is 'row':
            return int(np.arange(0,len(self))[observation_count==1])

        elif output is 'value':
            return first_observation.values[0]


    def calculate_total_return(self):
        '''
        Calculates the total return of a financial series.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'type needs to be price, return, or logreturn'
        gross_returns = self.to_returns()+1
        total_return = gross_returns.prod()-1
        return total_return


    def calculate_t_statistic(self, hypothesis=0, alpha=None):
        '''
        Returns the t-statistic for the mean of returns given a hypothesis.
        '''
        returns = self.to_returns()
        sigma2 = returns.var()
        mu = returns.mean()
        num_obs = returns.count()

        t_statistic = (mu-hypothesis)/np.sqrt(sigma2/num_obs)

        if alpha is None:
            return t_statistic

        else:
            df = num_obs-1
            significant = abs(t_statistic) >= sp.stats.t.ppf(1-alpha/2, df=df)
            return t_statistic, significant


    def winsorise_returns(self, alpha=0.05, *args, **kwargs):
        '''
        Winsorises a return series.
        '''
        obstype = self.obstype
        assert obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        lower = returns.quantile(alpha)
        upper = returns.quantile(1-alpha)
        winsorised = returns.clip(lower, upper)
        winsorised.obstype = 'return'
        return winsorised.to_obstype(obstype, *args, **kwargs)


    def calculate_annual_arithmetic_return(self, annual_obs):
        '''
        Calculates the annualised arithmetic return.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        arithmetic_return = returns.mean() * annual_obs
        return arithmetic_return


    def calculate_annual_geometric_return(self, annual_obs):
        '''
        Calculates the annualised geometic return.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        gross_returns = self.to_returns()+1
        years = annual_obs/gross_returns.count()
        geometric_return = gross_returns.prod()**years-1
        return geometric_return


    def calculate_annual_volatility(self, annual_obs):
        '''
        Calculates the annualised volatility.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        volatility = returns.std() * np.sqrt(annual_obs)
        return volatility


    def calculate_gain_percentage(self):
        '''
        Calculates the annualised volatility.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        gain_percentage = (returns>0).sum()/returns.count()
        return gain_percentage


    def calculate_sharpe_ratio(self, annual_obs, return_type='geometric'):
        '''
        Calculates the Shape ratio.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        assert return_type in ['geometric', 'arithmetic'],\
            'return_type needs to be geometric or arithmetic'

        if return_type is 'geometric':
            ret = self.calculate_annual_geometric_return(annual_obs)
        else:
            ret = self.calculate_annual_arithmetic_return(annual_obs)
        volatility = self.calculate_annual_volatility(annual_obs)
        sharpe_ratio = ret/volatility
        return sharpe_ratio


    def calculate_downside_volatility(self, annual_obs=1):
        '''
        Calculates the downside volatility.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        mean_return = returns.mean()
        downside_returns = returns[returns<mean_return]-mean_return
        downside_var = (downside_returns**2).sum()/(downside_returns.count()-1)
        downside_volatility = np.sqrt(downside_var)*np.sqrt(annual_obs)
        return downside_volatility


    def calculate_max_drawdown(self):
        '''
        Calculates the maximum drawdown.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        prices = self.to_prices()
        max_drawdown = (prices/prices.cummax()).min()-1
        return max_drawdown


    def calculate_historic_value_at_risk(self, confidence=0.95, period_obs=1):
        '''
        Calculates the historic VaR.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        value_at_risk = returns.quantile(1-confidence)
        return value_at_risk*period_obs


    def calculate_parametric_value_at_risk(self, confidence=0.95, period_obs=1):
        '''
        Calculates the parametric VaR.
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        mu = returns.mean()
        sigma = returns.std()
        value_at_risk = sp.stats.norm.ppf(1-confidence, loc=mu, scale=sigma)
        return value_at_risk*period_obs


    def calculate_historic_expected_shortfall(self, confidence=0.95, period_obs=1):
        '''
        Calculates the historic expected shortfall (conditional VaR).
        '''
        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        cutoff = returns.quantile(1-confidence)
        conditional_returns = returns[returns<cutoff]
        expected_shortfall = conditional_returns.mean()
        return expected_shortfall*period_obs


    def summarise_performance(self, annual_obs=1):
        '''
        Summarises the performance of a financial series.
        '''
        summary = pd.Series(dtype='float')
        summary = summary.append(pd.Series({'Return p.a. (arithmetic)': \
            self.calculate_annual_arithmetic_return(annual_obs)}))
        summary = summary.append(pd.Series({'Return p.a. (geometric)': \
            self.calculate_annual_geometric_return(annual_obs)}))
        summary = summary.append(pd.Series({'Volatility p.a.': \
            self.calculate_annual_volatility(annual_obs)}))
        summary = summary.append(pd.Series({'Sharpe ratio': \
            self.calculate_sharpe_ratio(annual_obs)}))
        summary = summary.append(pd.Series({'t-stat': \
            self.calculate_t_statistic()}))
        summary = summary.append(pd.Series({'Total return': \
            self.calculate_total_return()}))
        summary = summary.append(pd.Series({'Positive returns %': \
            self.calculate_gain_percentage()*100}))
        summary = summary.append(pd.Series({'VaR 95% (historic)': \
            self.calculate_historic_value_at_risk()}))
        summary = summary.append(pd.Series({'VaR 95% (parametric)': \
            self.calculate_parametric_value_at_risk()}))
        summary = summary.append(pd.Series({'Expected shortfall 95%': \
            self.calculate_historic_expected_shortfall()}))
        summary = summary.append(pd.Series({'Downside volatility': \
            self.calculate_downside_volatility()}))
        summary = summary.append(pd.Series({'Maximum drawdown': \
            self.calculate_max_drawdown()}))
        return summary


    ## # TODO: calculate_certainty_equivalent, estimate_factor_model
    # calculate_idiosyncratic_volatility


    ## DEPRECATED CODE


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
        assert self.obstype is 'logreturn', 'obstype is not logreturn'
        prices = np.exp(self)
        prices.iat[0] = init_price
        prices = prices.cumprod()
        prices.obstype = 'price'
        return prices


    def convert_logreturns_to_returns(self):
        '''
        Converts a financial series from observation type return to logreturn.
        '''
        assert self.obstype is 'logreturn', 'obstype is not logreturn'
        returns = np.exp(self)-1
        returns.obstype = 'return'
        return returns
