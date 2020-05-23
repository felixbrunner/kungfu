import numpy as np
import pandas as pd

import statsmodels.api as sm
import linearmodels as lm

from kungfu.series import FinancialSeries

'''
TO DO:
- needs to store FinancialSeries obstype
- make series methods available
- Calculate GRS test on factormodel class
'''


class FinancialDataFrame(pd.DataFrame):

    '''
    A Financial Data Frame is an extension of a pandas DataFrame that contains
    financial observations.
    '''

    _attributes_ = 'obstypes'

    # constructor properties to output the right types

    #attribute copy causes error for FinancialDataFrame (cause: _from_axes)
    #https://github.com/pandas-dev/pandas/issues/19300

    #@property
    #def _constructor(self):
    #    def construct(*args, **kw):
    #        fdf = FinancialDataFrame(*args, **kw)
    #        self._copy_attributes(fdf)
    #        return fdf
    #    return construct

    @property
    def _constructor(self):
        return FinancialDataFrame

    @property
    def _constructor_sliced(self):
        return FinancialSeries


    def __init__(self, *args, **kwargs):
        super(FinancialDataFrame, self).__init__(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], FinancialDataFrame):
            args[0]._copy_attributes(self)
        self.obstypes = None


    def _copy_attributes(self, fdf):

        '''
        Helps to keep attributes attached to instances of FinancialDataFrame to
        still be attached to the output when callung standard pandas methods on
        DataFrame.
        '''

        for attribute in self._attributes_.split(","):
            fdf.__dict__[attribute] = getattr(self, attribute)


    def standardise_values(self, axis=0, loc=0, scale=1):

        '''
        Standardises dataframe along input dimension.
        Standardisation involves subtracting the mean and dividing by the
        standard deviation.
        '''

        mus = self.mean(axis=axis)+loc
        sigmas = self.std(axis=axis)/scale

        other_axis = int(not axis)

        fdf_standardised = self.subtract(mus, axis=other_axis)\
                               .divide(sigmas, axis=other_axis)
        return fdf_standardised


    def export_to_latex(self, filename='financialdataframe.tex',
                        path=None, **kwargs):

        '''
        Exports FinancialDataFrame to LaTex format and saves it to a tex file
        using standard configuration settings.
        '''

        if filename[-4:] != '.tex':
            filename += '.tex'
        buf = path+filename
        self.to_latex(buf=buf, multirow=False, multicolumn_format ='c',\
                        na_rep='', escape=False, **kwargs)


    def fit_linear_regression(self, endog, exog, constant=True,
            lag=0, missing='drop', **kwargs):

        '''
        Run an OLS regression on selected columns of the FinancialDataFrame.
        endog and exog should be str (or list of str) to corresponding to column
        names.

        TO DO: interactions
        '''

        y = self[endog]
        if constant:
            X = sm.add_constant(self[exog]).shift(lag)
        else:
            X = self[exog].shift(lag)
        model = sm.OLS(y, X, missing=missing, **kwargs).fit()
        return model


    def fit_panel_regression(self, endog, exog, fixed_effects=[], lag=0,
            constant=True, cov_type='kernel', **kwargs):

        '''
        Run a panel regression on selected columns of the FinancialDataFrame.
        endog and exog should be str (or list of str) to corresponding to column
        names.
        fixed_effects should be a list of 'time' and/or 'entity' to define fixed
        effects.
        A constant will be automatically omitted if any fixed effects are
        included.
        lag defines the number of lags to use for the independent variable.
        '''

        assert type(self.index) is pd.MultiIndex,\
            'No panel data found, use fit_panel_regression instead'

        y = self[endog]
        if fixed_effects==[] and constant:
            X = sm.add_constant(self[exog])\
                .groupby(self.index.get_level_values(0)).shift(lag)
        else:
            X = self[exog]\
                .groupby(self.index.get_level_values(0)).shift(lag)

        entity_effects = 'entity' in fixed_effects
        time_effects ='time' in fixed_effects

        model = lm.PanelOLS(y, X, time_effects=time_effects,
                            entity_effects=entity_effects, **kwargs)\
                            .fit(cov_type=cov_type)
        return model


    def sort_portfolios(self, return_column, sorting_columns, n_sorts=10, lag=1,
                    method='simultaneous', **kwargs):

        '''
        Sort returns into portfolios based on one or more sorting variables.
        Method can be simultaneous or sequential.
        '''

        import kungfu.portfolio as portfolio

        return_data = self[return_column]
        sorting_data = self[sorting_columns]

        results = portfolio.sort_portfolios(return_data, sorting_data,
                        n_sorts=n_sorts, lag=lag,  method=method, **kwargs)
        return results


    def summarise_performance(self, obstype='return', annual_obs=1):

        '''
        Summarises the performance of each FinancialSeries in the
        FinancialDataFrame.
        obstype needs to be consistent across columns.
        '''

        summary = FinancialDataFrame(columns=self.columns)
        for (asset_name, asset_data) in self.iteritems():
            summary.loc[:,asset_name] = asset_data\
                                            .set_obstype(obstype)\
                                            .summarise_performance(annual_obs)
        return summary
