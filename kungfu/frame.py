import numpy as np
import pandas as pd

from kungfu.series import FinancialSeries


class FinancialDataFrame(pd.DataFrame):

    '''
    A Financial Data Frame is a pandas DataFrame that contains financial observations.
    '''

    _attributes_ = "obstypes"

    # constructor properties to output the right types
    @property
    def _constructor(self):
        return FinancialDataFrame

    @property
    def _constructor_sliced(self):
        return FinancialSeries


    def __init__(self, *args, **kwargs):
        super(FinancialDataFrame, self).__init__(*args, **kwargs)


## TODO: needs to store FinancialSeries obstype
