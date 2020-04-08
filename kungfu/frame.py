import numpy as np
import pandas as pd

from kungfu.series import FinancialSeries


class FinancialDataFrame(pd.DataFrame):

    '''
    A Financial Data Frame is a pandas DataFrame that contains financial observations.
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


## TODO: needs to store FinancialSeries obstype
## make series methods available
