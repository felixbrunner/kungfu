import scipy as sp
import pandas as pd
import numpy as np
import warnings


class FinancialSeries(pd.Series):

    '''
    A FinancialSeries is a pandas Series that contains financial observations.

    TO DO:
    - estimate factor model
    - calculate_idiosyncratic_volatility
    - realised volatility from returns

    NOTE:
    https://github.com/pandas-dev/pandas/pull/28573 needs to be resolved for
    pd.groupby to work with class methods
    '''

    _attributes_ = "obstype"

    # constructor properties to output the right types
    @property
    def _constructor(self):
        def construct(*args, **kwargs):
            fs = FinancialSeries(*args, **kwargs)
            self._copy_attributes(fs)
            return fs
        return construct

    @property
    def _constructor_expanddim(self):
        from kungfu.frame import FinancialDataFrame
        return FinancialDataFrame


    def __init__(self, *args, **kwargs):
        super(FinancialSeries, self).__init__(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], FinancialSeries):
            args[0]._copy_attributes(self)
        self.obstype = None


    def _copy_attributes(self, fs):

        '''
        Helps to keep attributes attached to instances of FinancialSeries to
        still be attached to the output when callung standard pandas methods on
        Series.
        '''

        for attribute in self._attributes_.split(","):
            fs.__dict__[attribute] = getattr(self, attribute)


    def set_obstype(self, obstype):

        '''
        Sets FinancialSeries attribute obstype.
        Needs to be in price, return, logreturn, characteristic, or None.
        '''

        assert obstype in ['price',
                           'return',
                           'logreturn',
                           'characteristic',
                           None],\
            'obstype needs to be return, price, logreturn, or characteristic'
        self.obstype = obstype
        return self


    def to_returns(self):

        '''
        Converts a FinancialSeries of obstype price, return or logreturns to
        obstype return.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype is not price, return, or logreturn'

        if self.obstype is 'return':
            return self

        elif self.obstype is 'price':
            returns = self.pct_change()

        elif self.obstype is 'logreturn':
            returns = np.exp(self)-1

        returns.obstype = 'return'
        return returns


    def to_logreturns(self):

        '''
        Converts a FinancialSeries of obstype price, return or logreturns to
        obstype logreturn.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype is not price, return, or logreturn'

        if self.obstype is 'logreturn':
            return self

        elif self.obstype is 'price':
            logreturns = np.log(self) - np.log(self.shift(periods=1))

        elif self.obstype is 'return':
            logreturns = np.log(self+1)

        logreturns.obstype = 'logreturn'
        return logreturns


    def to_prices(self, init_price=1):

        '''
        Converts a FinancialSeries of obstype price, return or logreturns to
        obstype prices.
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

        elif self.obstype is 'return':
            prices = self+1
            start_index = self.find_first_observation(output='row')-1
            prices.iat[start_index] = init_price
            prices = prices.cumprod()

        elif self.obstype is 'logreturn':
            prices = np.exp(self)
            start_index = self.find_first_observation(output='row')-1
            prices.iat[start_index] = init_price
            prices = prices.cumprod()

        prices.obstype = 'price'
        return prices


    def to_obstype(self, type, *args, **kwargs):

        '''
        Converts a FinancialSeries of obstype price, return or logreturns to
        input obstype.
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
        Converts a FinancialSeries of obstype price, return or logreturns into
        an index starting from 100.
        '''

        return self.to_prices(init_price=100)


    def to_cumreturns(self):

        '''
        Converts a FinancialSeries of obstype price, return or logreturns to
        cumulative returns.
        '''

        return self.to_prices(init_price=1)-1


    def find_first_observation(self, output='full'):

        '''
        Finds the first available observation and returns it with its index.
        Inputs:
        output - full, index, row, or value
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
        Calculates the total return of a FinancialSeries of obstype price,
        return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'type needs to be price, return, or logreturn'
        gross_returns = self.to_returns()+1
        total_return = gross_returns.prod()-1
        return total_return


    def calculate_t_statistic(self, hypothesis=0, alpha=None):

        '''
        Returns the t-statistic for the mean of returns given a hypothesis.
        Input needs to be of obstype price, return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'type needs to be price, return, or logreturn'
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
        Input needs to be of obstype price, return or logreturns.
        '''

        obstype = self.obstype
        assert obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        lower = returns.quantile(alpha)
        upper = returns.quantile(1-alpha)
        winsorised = returns.clip(lower, upper)
        return winsorised.to_obstype(obstype, *args, **kwargs)


    def calculate_annual_arithmetic_return(self, annual_obs):

        '''
        Calculates the annualised arithmetic return for a FinancialSeries of
        obstype price, return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        arithmetic_return = returns.mean() * annual_obs
        return arithmetic_return


    def calculate_annual_geometric_return(self, annual_obs):

        '''
        Calculates the annualised geometic return for a FinancialSeries of
        obstype price, return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        gross_returns = self.to_returns()+1
        years = annual_obs/gross_returns.count()
        geometric_return = gross_returns.prod()**years-1
        return geometric_return


    def calculate_annual_volatility(self, annual_obs):

        '''
        Calculates the annualised volatility for a FinancialSeries of obstype
        price, return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        volatility = returns.std() * np.sqrt(annual_obs)
        return volatility


    def calculate_gain_percentage(self):

        '''
        Calculates the annualised volatility for a FinancialSeries of obstype
        price, return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        gain_percentage = (returns>0).sum()/returns.count()
        return gain_percentage


    def calculate_sharpe_ratio(self, annual_obs, return_type='geometric'):

        '''
        Calculates the Shape ratio for a FinancialSeries of obstype price,
        return or logreturns.
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
        Calculates the downside volatility for a FinancialSeries of obstype
        price, return or logreturns..
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
        Calculates the maximum drawdown for a FinancialSeries of obstype price,
        return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        prices = self.to_prices()
        max_drawdown = (prices/prices.cummax()).min()-1
        return max_drawdown


    def calculate_historic_value_at_risk(self, confidence=0.95, period_obs=1):

        '''
        Calculates the historic VaR for a FinancialSeries of obstype price,
        return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        value_at_risk = returns.quantile(1-confidence)
        return value_at_risk*period_obs


    def calculate_parametric_value_at_risk(self, confidence=0.95, period_obs=1):

        '''
        Calculates the parametric VaR for a FinancialSeries of obstype price,
        return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        alpha = 1-confidence
        mu = returns.mean()
        sigma = returns.std()
        value_at_risk = mu+sigma*sp.stats.norm.ppf(alpha)
        return value_at_risk*period_obs


    def calculate_historic_expected_shortfall(self, confidence=0.95, period_obs=1):

        '''
        Calculates the historic expected shortfall (conditional VaR) for a
        FinancialSeries of obstype price, return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        cutoff = returns.quantile(1-confidence)
        conditional_returns = returns[returns<cutoff]
        expected_shortfall = conditional_returns.mean()
        return expected_shortfall*period_obs


    def calculate_parametric_expected_shortfall(self, confidence=0.95, period_obs=1):

        '''
        Calculates the historic expected shortfall (conditional VaR) for a
        FinancialSeries of obstype price, return or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        alpha = 1-confidence
        mu = returns.mean()
        sigma = returns.std()
        factor = sp.stats.norm.pdf(sp.stats.norm.ppf(alpha))
        expected_shortfall = mu-sigma/alpha*factor
        return expected_shortfall*period_obs


    def summarise_performance(self, annual_obs=1):

        '''
        Summarises the performance of a FinancialSeries of obstype price, return
        or logreturns.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
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


    def calculate_certainty_equivalent(self, utility_function, **kwargs):

        '''
        Returns the certainty equivalent of a FinancialSeries of obstype price,
        return or logreturns, given a utility function.
        Inputs:
        utility_function - a function handle that defines utility give a return
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()
        expected_utility = returns.apply(utility_function, **kwargs).mean()
        utility_difference = lambda r : utility_function(r)-expected_utility
        certainty_equivalent = sp.optimize.fsolve(utility_difference,0)
        return certainty_equivalent[0]


    def standardise_values(self, loc=0, scale=1, return_params=False):

        '''
        Standardises FinancialSeries by subtracting the mean and dividing by the
        standard deviation.
        '''

        mu = self.mean()+loc
        sigma = self.std()*scale
        fs_standardised = self.subtract(mu).divide(sigma)
        if return_params:
            return fs_standardised, (mu, sigma)
        else:
            return fs_standardised


    def shrink_outliers(self, alpha=0.05, lamb=1):

        '''
        This function shrinks outliers in a series towards the threshold values.
        The parameter alpha defines the threshold values as a multiple of one sample standard deviation.
        The parameter lamb defines the degree of shrinkage of outliers towards the thresholds.

        The transformation is as follows:
        if the z score is inside the thresholds f(x)=x
        if it is above the upper threshold f(x)=1+1/lamb*ln(x+(1-lamb)/lamb)-1/lamb*ln(1/lamb)
        if it is below the lower threshold f(x)=-1-1/lamb*ln(-x+(1-lamb)/lamb)+1/lamb*ln(1/lamb)
        '''

        mu = self.mean()
        sigma = self.std()
        valid = self.notna()
        z_scores = self[valid].standardise_values()
        threshold = abs(sp.stats.norm.ppf(alpha))

        adjusted_scores = z_scores/threshold
        adjusted_scores[adjusted_scores.values>1] = \
            1+1/lamb*np.log(adjusted_scores[adjusted_scores.values>1]+ \
            (1-lamb)/lamb)-1/lamb*np.log(1/lamb)
        adjusted_scores[adjusted_scores.values<-1] = \
            -1-1/lamb*np.log((1-lamb)/lamb-adjusted_scores[adjusted_scores.values<-1])+ \
            1/lamb*np.log(1/lamb)

        shrank_z_scores = adjusted_scores*threshold
        shrank_series = self.copy()
        shrank_series[valid] = shrank_z_scores*sigma+mu
        return shrank_series


    def fill_missing_prices_inside(self, method = 'ffill', limit = None):

        '''
        fills missing values in the middle of a FinancialSeries
        (but not at beginning and end)
        Inputs:
        meth - zero, ffill, bfill
        limit - integer
        '''

        obstype = self.obstype
        assert obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        if obstype is not 'price':
            prices = self.to_prices()
        else:
            prices = self

        noprice = (prices.bfill().isna() | prices.ffill().isna())
        if method is 'zero':
            filled = prices.fillna(0)
        else:
            filled = prices.fillna(method = method, limit = limit)
        filled[noprice] = np.nan
        if obstype is not 'price':
            filled = filled.to_obstype(obstype)
        return filled


    def calculate_realised_volatility(self, annual_obs=1):

        '''
        Calculates realised volatility for a FinancialSeries of obstype price,
        return or logreturn from squared returns.
        If annual_obs is input, the result will be annualised.
        '''

        assert self.obstype in ['price','return','logreturn'],\
            'obstype needs to be price, return, or logreturn'
        returns = self.to_returns()

        realised_volatility = ((returns**2).mean())**0.5 * np.sqrt(annual_obs)
        return realised_volatility


    def standardise_values(self, loc=0, scale=1):

        '''
        Standardises FinancialSeries involving subtracting the mean and dividing by
        the standard deviation.
        '''

        mu = self.mean()+loc
        sigma = self.std()/scale
        fs_standardised = self.subtract(mu).divide(sigma)
        return fs_standardised


    def create_index(self, weighting_data=None, lag=0, **kwargs):

        '''
        Returns a FinancialSeries that contains returns of an equal or weighted
        index.
        Weights sum up to one in each period.
        '''

        import kungfu.index as index
        index_returns = index.create_index(return_data=self,
                            weighting_data=weighting_data, lag=lag, **kwargs)

        return index_returns
