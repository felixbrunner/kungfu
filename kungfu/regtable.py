import pandas as pd
import numpy as np
import statsmodels.api as sm
import linearmodels as lm


def create_statsmodels_summary_column(regression, t_stats=True, add_outputs=[]):

    '''
    Creates a pandas series object that contains the formatted results
    of a statsmodels regression for academic presentation.
    Additional outputs can be:
        - R-squared
        - N
        - Adj R-squared
        - AIC
        - BIC
        - LL
        - F-stat
        - P(F-stat)
        - DF (model)
        - DF (residuals)
        - MSE (model)
        - MSE (residuals)
        - MSE (total)
    '''

    summary = pd.Series(index=pd.MultiIndex.from_product([regression.params.index, ['coeff', 't-stat']]))
    stars = (abs(regression.tvalues) > 1.645).astype(int) + \
            (abs(regression.tvalues) > 1.96).astype(int) + \
            (abs(regression.tvalues) > 2.58).astype(int)
    summary.loc[(regression.params.index, 'coeff')] = \
        regression.params.map(lambda x: '%.4f' % x).values + \
        stars.map(lambda x: x * '*').values

    if t_stats:
        summary.loc[(regression.params.index, 't-stat')] = \
            regression.tvalues.map(lambda x: '(%.4f)' % x).values
    else:
        summary.index = pd.MultiIndex.from_product([regression.params.index, ['coeff', 's.e.']])
        summary.loc[(regression.params.index, 's.e.')] = \
            regression.bse.map(lambda x: '(%.4f)' % x).values

    output_dict = {'R-squared': 'regression.rsquared',
                   'N': 'regression.nobs',
                   'Adj R-squared': 'regression.rsquared_adj',
                   'AIC': 'regression.aic',
                   'BIC': 'regression.bic',
                   'LL': 'regression.llf',
                   'F-stat': 'regression.fvalue',
                   'P(F-stat)': 'regression.f_pvalue',
                   'DF (model)': 'regression.df_model',
                   'DF (residuals)': 'regression.df_resid',
                   'MSE (model)': 'regression.mse_model',
                   'MSE (residuals)': 'regression.mse_resid',
                   'MSE (total)': 'regression.mse_total',
                   }

    for out in add_outputs:
        if out in ['N', 'DF (model)', 'DF (residuals)']:
            summary[(out, '')] = "{:.0f}".format(eval(output_dict[out]))
        else:
            try:
                summary[(out, '')] = "{:.4f}".format(eval(output_dict[out]))
            except:
                pass

    return summary


def create_linearmodels_summary_column(regression, t_stats=True, add_outputs=[]):

    '''
    Creates a pandas series object that contains the formatted results
    of a linearmodels regression for academic presentation.
    Additional outputs can be:
        - R-squared
        - N
        - R-squared (between)
        - R-squared (inclusive)
        - R-squared (overall)
        - R-squared (within)
        - LL
        - F-stat
        - P(F-stat)
        - F-stat (robust)
        - P(F-stat) (robust)
        - DF (model)
        - DF (residuals)
        - Time FE
        - Entity FE
        - Other FE
    '''

    summary = pd.Series(index=pd.MultiIndex.from_product([regression.params.index, ['coeff', 't-stat']]))
    stars = (abs(regression.tstats) > 1.645).astype(int) + \
            (abs(regression.tstats) > 1.96).astype(int) + \
            (abs(regression.tstats) > 2.58).astype(int)
    summary.loc[(regression.params.index, 'coeff')] = \
        regression.params.map(lambda x: '%.4f' % x).values + \
        stars.map(lambda x: x * '*').values

    if t_stats:
        summary.loc[(regression.params.index, 't-stat')] = \
            regression.tstats.map(lambda x: '(%.4f)' % x).values
    else:
        summary.index = pd.MultiIndex.from_product([regression.params.index, ['coeff', 's.e.']])
        summary.loc[(regression.params.index, 's.e.')] = \
            regression.std_errors.map(lambda x: '(%.4f)' % x).values

    output_dict = {'R-squared': 'regression.rsquared',
                   'N': 'regression.nobs',
                   'R-squared (between)': 'regression.rsquared_between',
                   'R-squared (inclusive)': 'regression.rsquared_inclusive',
                   'R-squared (overall)': 'regression.rsquared_overall',
                   'R-squared (within)': 'regression.rsquared_within',
                   'LL': 'regression.loglik',
                   'F-stat':'regression.f_statistic.stat',
                   'P(F-stat)': 'regression.f_statistic.pval',
                   'F-stat (robust)': 'regression.f_statistic_robust.stat',
                   'P(F-stat) (robust)': 'regression.f_statistic_robust.pval',
                   'DF (model)': 'regression.df_model',
                   'DF (residuals)': 'regression.df_resid',
                   'Time FE': 'regression.model.time_effects',
                   'Entity FE': 'regression.model.entity_effects',
                   'Other FE': 'regression.model.other_effects',
                   }

    for out in add_outputs:
        if out in ['N', 'DF (model)', 'DF (residuals)']:
            summary[(out, '')] = "{:.0f}".format(eval(output_dict[out]))
        elif out in ['Time FE', 'Entity FE', 'Other FE']:
            summary[(out, '')] = str(eval(output_dict[out]))
        else:
            try:
                summary[(out, '')] = "{:.4f}".format(eval(output_dict[out]))
            except:
                pass

    return summary


class RegressionTable(pd.DataFrame):

    '''
    A Regression Table is a pandas dataframe that contains the outputs of
    regression models for presentation in an academic context.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nregs = self.shape[1]
        if len(self.index) == 0:
            self.index = pd.MultiIndex.from_product([[], []])


    @property
    def _constructor(self):
        return RegressionTable


    def join_regression(self, regression_model, **kwargs):
        self._nregs += 1
        if type(regression_model) is sm.regression.linear_model.RegressionResultsWrapper:
            column = create_statsmodels_summary_column(regression_model, **kwargs).rename('('+str(self._nregs)+')')
        elif type(regression_model) is lm.panel.results.PanelEffectsResults:
            column = create_linearmodels_summary_column(regression_model, **kwargs).rename('('+str(self._nregs)+')')
        joined_table = self.join(column, how='outer', sort=False)\
                           .replace(np.nan,'')
        top = joined_table[joined_table.index.get_level_values(1) != '']
        bottom = joined_table[joined_table.index.get_level_values(1) == '']
        return top.append(bottom)


    def export_to_latex(self, filename=None, **kwargs):
        if filename is None:
            filename = input('Specify a filename (e.g.: regression_table.tex):')
        if filename[-4:] != '.tex':
            filename += '.tex'
        self.to_latex(buf=filename, multirow=False, multicolumn_format='c', na_rep='', escape=False, **kwargs)


    def change_row_labels(self, index_dict):
        label_map = dict(zip(self.index.get_level_values(0),\
                             self.index.get_level_values(0)))
        label_map.update(index_dict)
        new_index = pd.MultiIndex.from_arrays([[label_map.get(i) for i in list(self.index.get_level_values(0))], \
                                               list(self.index.get_level_values(1))])
        self.index = new_index
        return self


    def change_column_labels(self, reg_dict):
        label_map = dict(zip(self.columns, self.columns))
        label_map.update(reg_dict)
        new_columns = pd.Index([label_map.get(i) for i in list(self.columns)])
        self.columns = new_columns
        return self


    def drop_second_index(self):
        new_index = list(regtable.index.get_level_values(0))
        hide_duplicate = [False]+[i==j for i,j in zip(new_index[1:], new_index[:-1])]
        new_index = ['' if h else i for i,h in zip(new_index,hide_duplicate)]
        self.index = new_index
        return self


    def change_variable_order(self, variable_list):
        top = self[self.index.get_level_values(1) != '']
        bottom = self[self.index.get_level_values(1) == '']
        second_level = list(top.index.get_level_values(1).unique())
        new_order = [(var,second) for var in variable_list for second in second_level]
        top = top.reindex(new_order)
        return top.append(bottom)
