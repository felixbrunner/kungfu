import numpy as np
import pandas as pd
import datetime as dt
import kungfu.dataloader as data
import matplotlib.pyplot as plt


def add_recession_bars(ax, freq='D', startdate='1/1/1900', enddate=dt.datetime.today()):

    '''
    Adds NBER recession bars to plotly axis of a timeseries plot.
    '''

    # get data
    usrec = data.download_recessions_data(freq=freq, startdate=startdate, enddate=enddate)

    # create list of recessions
    rec_start = usrec.diff(1)
    rec_end = usrec.diff(-1)
    rec_start.iloc[0] = usrec.iloc[0]
    rec_end.iloc[-1] = usrec.iloc[-1]
    rec_start_dates = rec_start.query('USREC==1')
    rec_end_dates = rec_end.query('USREC==1')

    # add recessions to matplotlib axis
    for i_rec in range(0,len(rec_start_dates)):
        ax.axvspan(rec_start_dates.index[i_rec], rec_end_dates.index[i_rec], color='grey', linewidth=0, alpha=0.4)

    # old version
    #usrec = usrec['USREC']
    #ax.fill_between(usrec.index, ax.get_ylim()[0], ax.get_ylim()[1], where=usrec.values, color='grey', alpha=0.4)



###################
## PLOT SETTINGS ##
###################

# colors
DARK_BLUE = '#014c63'
DARK_GREEN = '#3b5828'
DARK_RED = '#880000'
ORANGE = '#ae3e00'
DARK_YELLOW = '#ba9600'
PURPLE = '#663a82'
color_list = [DARK_BLUE, DARK_GREEN, DARK_RED, DARK_YELLOW,
              ORANGE, PURPLE]

plt.style.use('seaborn')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
plt.rcParams['figure.figsize'] = [17, 8]
