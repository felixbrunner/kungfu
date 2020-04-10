import numpy as np
import pandas as pd

from kungfu.series import FinancialSeries


class FinancialDataFrame(pd.DataFrame):

    '''
    A Financial Data Frame is a pandas DataFrame that contains financial
    observations.
    '''

    _attributes_ = "obstypes"

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





## TODO: needs to store FinancialSeries obstype
## make series methods available

## Calculate GRS test on factormodel class
