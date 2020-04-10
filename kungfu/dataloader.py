import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import requests
import io
import zipfile

from kungfu.series import FinancialSeries
from kungfu.frame import FinancialDataFrame


def download_factor_data(freq='D'):

    '''
    Downloads factor data from Kenneth French's website and returns dataframe.
    freq can be either 'D' (daily) or 'M' (monthly).
    '''

    if freq is 'D':
        # Download Carhartt 4 Factors
        factors_daily = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start='1/1/1900')[0]
        mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start='1/1/1900')[0]
        factors_daily = factors_daily.join(mom)
        factors_daily = factors_daily[['Mkt-RF','SMB','HML','Mom   ','RF']]
        factors_daily.columns = ['Mkt-RF','SMB','HML','Mom','RF']
        return FinancialDataFrame(factors_daily)

    elif freq is 'M':
        # Download Carhartt 4 Factors
        factors_monthly = web.DataReader("F-F_Research_Data_Factors", "famafrench", start='1/1/1900')[0]
      #  mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start='1/1/1900')[0] #There seems to be a problem with the data file, fix if mom is needed
      #  factors_monthly = factors_monthly.join(mom)
      #  factors_monthly = factors_monthly[['Mkt-RF','SMB','HML','Mom   ','RF']]
        factors_monthly.index = factors_monthly.index.to_timestamp()
      #  factors_monthly.columns = ['Mkt-RF','SMB','HML','Mom','RF']
        factors_monthly.columns = ['Mkt-RF','SMB','HML','RF']
        factors_monthly.index = factors_monthly.index+pd.tseries.offsets.MonthEnd(0)
        return FinancialDataFrame(factors_monthly)


def download_industry_data(freq='D', excessreturns = True):

    '''
    Downloads industry data from Kenneth French's website and returns dataframe.
    freq can be either 'D' (daily) or 'M' (monthly).
    excessreturns is a boolean to define if the the function should return excess returns.
    '''

    if freq is 'D':
        # Download Fama/French 49 Industries
        industries_daily = web.DataReader("49_Industry_Portfolios_Daily", "famafrench", start='1/1/1900')[0]
        industries_daily[(industries_daily <= -99.99) | (industries_daily == -999)] = np.nan #set missing data to NaN
        industries_daily = industries_daily.rename_axis('Industry', axis='columns')
        if excessreturns is True:
            factors_daily = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start='1/1/1900')[0]
            industries_daily = industries_daily.subtract(factors_daily['RF'], axis=0) #transform into excess returns
        return industries_daily

    elif freq is 'M':
        # Download Fama/French 49 Industries
        industries_monthly = web.DataReader("49_Industry_Portfolios", "famafrench", start='1/1/1900')[0]
        industries_monthly[(industries_monthly <= -99.99) | (industries_monthly == -999)] = np.nan #set missing data to NaN
        industries_monthly = industries_monthly.rename_axis('Industry', axis='columns')
        industries_monthly.index = industries_monthly.index.to_timestamp()
        if excessreturns is True:
            factors_monthly = web.DataReader("F-F_Research_Data_Factors", "famafrench", start='1/1/1900')[0]
            factors_monthly.index = factors_monthly.index.to_timestamp()
            industries_monthly = industries_monthly.subtract(factors_monthly['RF'], axis=0) #transform into excess returns
        industries_monthly.index = industries_monthly.index+pd.tseries.offsets.MonthEnd(0)
        return industries_monthly


def download_25portfolios_data(freq='D', excessreturns = True):

    '''
    Downloads 25 portfolios data from Kenneth French's website and returns dataframe.
    freq can be either 'D' (daily) or 'M' (monthly).
    excessreturns is a boolean to define if the the function should return excess returns.
    '''

    if freq is 'D':
        # Download Fama/French 25 portfolios
        portfolios_daily = web.DataReader("25_Portfolios_5x5_CSV", "famafrench", start='1/1/1900')[0]
        portfolios_daily[(portfolios_daily <= -99.99) | (portfolios_daily == -999)] = np.nan #set missing data to NaN
        if excessreturns is True:
            factors_daily = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start='1/1/1900')[0]
            portfolios_daily = portfolios_daily.subtract(factors_daily['RF'], axis=0) #transform into excess returns
        return portfolios_daily

    elif freq is 'M':
        # Download Fama/French 25 portfolios
        portfolios_monthly = web.DataReader("25_Portfolios_5x5_Daily_CSV", "famafrench", start='1/1/1900')[0]
        portfolios_monthly[(industries_monthly <= -99.99) | (industries_monthly == -999)] = np.nan #set missing data to NaN
        portfolios_monthly.index = portfolios_monthly.index.to_timestamp()
        if excessreturns is True:
            factors_monthly = web.DataReader("F-F_Research_Data_Factors", "famafrench", start='1/1/1900')[0]
            factors_monthly.index = factors_monthly.index.to_timestamp()
            portfolios_monthly = portfolios_monthly.subtract(factors_monthly['RF'], axis=0) #transform into excess returns
        return portfolios_monthly


def download_recessions_data(freq='M', startdate='1/1/1900', enddate=dt.datetime.today()):

    '''
    Downloads NBER recessions from FRED and returns series.
    freq can be either 'D' (daily) or 'M' (monthly).
    startdate and enddate define the length of the timeseries.
    '''

    USREC_monthly = web.DataReader('USREC', 'fred',start = startdate, end=enddate)
    if freq is 'M':
        return USREC_monthly

    if freq is 'D':
        first_day = USREC_monthly.index.min() - pd.DateOffset(day=1)
        last_day = USREC_monthly.index.max() + pd.DateOffset(day=31)
        dayindex = pd.date_range(first_day, last_day, freq='D')
        dayindex.name = 'DATE'
        USREC_daily = USREC_monthly.reindex(dayindex, method='ffill')
        return USREC_daily


def download_jpy_usd_data():

    '''
    Downloads USD/JPY exchange rate data from FRED and returns series.
    '''

    jpy = web.DataReader('DEXJPUS', 'fred', start = '1900-01-01')
    return jpy


def download_cad_usd_data():

    '''
    Downloads USD/CAD exchange rate data from FRED and returns series.
    '''

    cad = web.DataReader('DEXCAUS', 'fred', start = '1900-01-01')
    return cad


def download_vix_data():

    '''
    Downloads VIX index data from FRED and returns series.
    '''

    vix = web.DataReader('VIXCLS', 'fred', start = '1900-01-01')
    return vix


def download_goyal_welch_svar():

    '''
    Downloads Goyal/Welch SVAR data from Amit Goyal's website and returns DataFrame.
    '''

    url = 'http://www.hec.unil.ch/agoyal/docs/PredictorData2017.xlsx'
    sheet = pd.read_excel(url, sheet_name='Monthly')
    dates = sheet['yyyymm']
    SVAR = pd.DataFrame(sheet['svar'])
    SVAR.index = [(dt.datetime(year = math.floor(date/100),month = date%100,day = 1)+dt.timedelta(days=32)).replace(day=1)-dt.timedelta(days=1) for date in dates]
    return SVAR


def download_sadka_liquidity():

    '''
    Downloads Sadka liquidity factor data from Ronnie Sadka's website and returns DataFrame.
    '''

    url = 'http://www2.bc.edu/ronnie-sadka/Sadka-LIQ-factors-1983-2012-WRDS.xlsx'
    sheet = pd.read_excel(url, sheet_name='Sheet1')
    dates = sheet['Date']
    SadkaLIQ1 = pd.DataFrame(sheet['Fixed-Transitory'])
    SadkaLIQ1.index = [(dt.datetime(year = math.floor(date/100),month = date%100,day = 1)+dt.timedelta(days=32)).replace(day=1)-dt.timedelta(days=1) for date in dates]
    SadkaLIQ2 = pd.DataFrame(sheet['Variable-Permanent'])
    SadkaLIQ2.index = [(dt.datetime(year = math.floor(date/100),month = date%100,day = 1)+dt.timedelta(days=32)).replace(day=1)-dt.timedelta(days=1) for date in dates]
    return SadkaLIQ1, SadkaLIQ2


def download_manela_kelly_he_intermediary():

    '''
    Downloads Manela/Kelly/He intermediary risk factor data from Manela's website and returns DataFrame.
    '''

    url = 'http://apps.olin.wustl.edu/faculty/manela/hkm/intermediarycapitalrisk/He_Kelly_Manela_Factors.zip'
    filename = 'He_Kelly_Manela_Factors_monthly.csv'
    column1 = 'intermediary_capital_ratio'
    column2 = 'intermediary_capital_risk_factor'
    column3 = 'intermediary_value_weighted_investment_return'
    column4 = 'intermediary_leverage_ratio_squared'
    raw_data = pd.read_csv(zipfile.ZipFile(io.BytesIO(requests.get(url).content)).open(filename))
    Intermediary = pd.DataFrame(raw_data[[column1,column2,column3,column4]]) #HeKellyManela
    dates = raw_data['yyyymm']
    Intermediary.index = [(dt.datetime(year = math.floor(date/100),month = date%100,day = 1)+dt.timedelta(days=32)).replace(day=1)-dt.timedelta(days=1) for date in dates]
    return Intermediary


def download_jln_macro_uncertainty():

    '''
    Downloads Jurado/Ludvigson/Ng macro uncertainty data from Sydney Ludvigson's website and returns DataFrame.
    '''

    url = 'https://www.sydneyludvigson.com/s/MacroFinanceUncertainty_201908_update.zip'
    filename = 'MacroUncertaintyToCirculate.csv'
    uncertainty = pd.read_csv(zipfile.ZipFile(io.BytesIO(requests.get(url).content)).open(filename), index_col='Date')
    uncertainty.index = pd.to_datetime(uncertainty.index)
    uncertainty.index = pd.DatetimeIndex([dt.datetime(year=i.year,month=i.month+1 if i.month<12 else 1,day=1) for i in uncertainty.index]) + dt.timedelta(days=-1)
    return uncertainty


def download_jln_real_uncertainty():

    '''
    Downloads Jurado/Ludvigson/Ng real uncertainty data from Sydney Ludvigson's website and returns DataFrame.
    '''

    url = 'https://www.sydneyludvigson.com/s/MacroFinanceUncertainty_201908_update.zip'
    filename = 'RealUncertaintyToCirculate.csv'
    uncertainty = pd.read_csv(zipfile.ZipFile(io.BytesIO(requests.get(url).content)).open(filename), index_col='Date')
    uncertainty.index = pd.to_datetime(uncertainty.index)
    uncertainty.index = pd.DatetimeIndex([dt.datetime(year=i.year,month=i.month+1 if i.month<12 else 1,day=1) for i in uncertainty.index]) + dt.timedelta(days=-1)
    return uncertainty


def download_jln_financial_uncertainty():

    '''
    Downloads Jurado/Ludvigson/Ng financial uncertainty data from Sydney Ludvigson's website and returns DataFrame.
    '''

    url = 'https://www.sydneyludvigson.com/s/MacroFinanceUncertainty_201908_update.zip'
    filename = 'FinancialUncertaintyToCirculate.csv'
    uncertainty = pd.read_csv(zipfile.ZipFile(io.BytesIO(requests.get(url).content)).open(filename), index_col='Date')
    uncertainty.index = pd.to_datetime(uncertainty.index)
    uncertainty.index = pd.DatetimeIndex([dt.datetime(year=i.year,month=i.month+1 if i.month<12 else 1,day=1) for i in uncertainty.index]) + dt.timedelta(days=-1)
    return uncertainty


def download_pastor_stambaugh_liquidity():

    '''
    #CURRENTLY NOT WORKING
    Downloads Stambaugh liquidity factor data from Pastor Stambaugh's website and returns DataFrame.
    '''

    url = 'https://faculty.chicagobooth.edu/lubos.pastor/research/liq_data_1962_2017.txt'
    #PSLIQ = pd.read_csv(url, sep=' ')
    PSLIQ = pd.read_csv(url, delim_whitespace=True, header=None)
    return PSLIQ
