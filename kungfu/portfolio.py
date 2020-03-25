import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt



def sort_portfolios(returns, ranking_variable, n_portfolios, lags=1, return_assets=False):
    # align periods
    sorting_variable = ranking_variable.shift(lags)

    # set up parameters
    [t,n] = returns.shape
    include = returns.notna() & sorting_variable.notna()
    n_period = include.sum(axis=1)

    # sort assets
    returns_include = returns[include]
    sorting_variable[~include] = np.nan
    cutoff_ranks = np.dot(n_period.values.reshape(t,1)/n_portfolios,np.arange(n_portfolios+1).reshape(1,n_portfolios+1)).round()
    asset_ranks = sorting_variable.rank(axis=1)

    # set up output frames
    portfolio_returns = pd.DataFrame(index=returns.index,columns=range(1,n_portfolios+1))
    portfolio_assets = pd.DataFrame(index=returns.index,columns=range(1,n_portfolios+1))
    portfolio_mapping = pd.DataFrame(index=returns.index, columns=returns.columns)

    # calculate outputs
    for i_portfolio in range(0,n_portfolios):
        lower = cutoff_ranks[:,i_portfolio].reshape(t,1).repeat(n, axis=1)
        upper = cutoff_ranks[:,i_portfolio+1].reshape(t,1).repeat(n, axis=1)
        portfolio_returns[i_portfolio+1] = returns_include[(asset_ranks>lower) & (asset_ranks<=upper)].mean(axis=1)
        portfolio_assets[i_portfolio+1] = ((asset_ranks>lower) & (asset_ranks<=upper)).sum(axis=1)
        portfolio_mapping[(asset_ranks>lower) & (asset_ranks<=upper)] = i_portfolio

    # outputs
    if return_assets == False:
        return portfolio_returns
    else:
        return portfolio_returns, portfolio_assets, portfolio_mapping



def double_sort_portfolios(returns, ranking_variable_1, ranking_variable_2, n_portfolios_1, n_portfolios_2, lags_1=1, lags_2=1, return_assets=False):
    # identify missing values
    exclude = returns.isna() | ranking_variable_1.shift(lags_1).isna() | ranking_variable_2.shift(lags_2).isna()
    returns[exclude] = np.nan

    # first sort
    portfolio_mapping_1 = sort_portfolios(returns, ranking_variable_1, n_portfolios_1, lags_1, return_assets=True)[2]

    # second sorts
    portfolio_mapping_2 = pd.DataFrame(0, index=portfolio_mapping_1.index, columns=portfolio_mapping_1.columns)
    for i_portfolio_2 in range(0,n_portfolios_2):
        subportfolio_returns = returns[portfolio_mapping_1 == i_portfolio_2]
        portfolio_mapping_2 += (sort_portfolios(subportfolio_returns, ranking_variable_2, n_portfolios_2, lags_2, return_assets=True)[2]).fillna(0)
    portfolio_mapping_2[exclude] = np.nan

    # combined sort
    portfolio_mapping = portfolio_mapping_1*n_portfolios_1 + portfolio_mapping_2

    # set up output frames
    portfolio_returns = pd.DataFrame(index=returns.index,columns=[str(i_portfolio_1+1)+','+str(i_portfolio_2+1) for i_portfolio_1 in range(0,n_portfolios_1) for i_portfolio_2 in range(0,n_portfolios_2)])
    portfolio_assets = pd.DataFrame(index=returns.index,columns=[str(i_portfolio_1+1)+','+str(i_portfolio_2+1) for i_portfolio_1 in range(0,n_portfolios_1) for i_portfolio_2 in range(0,n_portfolios_2)])

    # calculate outputs
    for i_portfolio_all in range(0,n_portfolios_1*n_portfolios_2):
        portfolio_returns.iloc[:,i_portfolio_all] = returns[portfolio_mapping == i_portfolio_all].mean(axis=1)
        portfolio_assets.iloc[:,i_portfolio_all] = (portfolio_mapping == i_portfolio_all).sum(axis=1)

    # outputs
    if return_assets == False:
        return portfolio_returns
    else:
        return portfolio_returns, portfolio_assets, portfolio_mapping



def double_sort_portfolios_simultaneously(returns, ranking_variable_1, ranking_variable_2, n_portfolios_1, n_portfolios_2, lags_1=1, lags_2=1, return_assets=False):
    # identify missing values
    exclude = returns.isna() | ranking_variable_1.shift(lags_1).isna() | ranking_variable_2.shift(lags_2).isna()
    returns[exclude] = np.nan

    # first sort
    portfolio_mapping_1 = sort_portfolios(returns, ranking_variable_1, n_portfolios_1, lags_1, return_assets=True)[2]

    # second sorts
    portfolio_mapping_2 = sort_portfolios(returns, ranking_variable_2, n_portfolios_2, lags_2, return_assets=True)[2]

    # combined sort
    portfolio_mapping = portfolio_mapping_1*n_portfolios_1 + portfolio_mapping_2

    # set up output frames
    portfolio_returns = pd.DataFrame(index=returns.index,columns=[str(i_portfolio_1+1)+','+str(i_portfolio_2+1) for i_portfolio_1 in range(0,n_portfolios_1) for i_portfolio_2 in range(0,n_portfolios_2)])
    portfolio_assets = pd.DataFrame(index=returns.index,columns=[str(i_portfolio_1+1)+','+str(i_portfolio_2+1) for i_portfolio_1 in range(0,n_portfolios_1) for i_portfolio_2 in range(0,n_portfolios_2)])

    # calculate outputs
    for i_portfolio_all in range(0,n_portfolios_1*n_portfolios_2):
        portfolio_returns.iloc[:,i_portfolio_all] = returns[portfolio_mapping == i_portfolio_all].mean(axis=1)
        portfolio_assets.iloc[:,i_portfolio_all] = (portfolio_mapping == i_portfolio_all).sum(axis=1)

    # outputs
    if return_assets == False:
        return portfolio_returns
    else:
        return portfolio_returns, portfolio_assets, portfolio_mapping
