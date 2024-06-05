# 2dfields.py>

# functions related to plotting data in the form of a table

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def make_table(rowlbls,collbls,data,ttl='',flnm=''):
    """
    Makes a simple table using matplotlibs default table

    Parameters
    ----------
    rowlbls : array of strings
        label of each row
    collbls : array of strings
        label of each column
    data : 2d array
        data to be plotted
    """
    fig, ax = plt.subplots()
    fig.set_figheight(.5*len(rowlbls))
    fig.set_figwidth(30)
    ax.set_axis_off()
    matplotlib.rcParams.update({'font.size': 10})
    table = ax.table(
        cellText = data,
        rowLabels = rowlbls,
        colLabels = collbls,
        rowColours =["palegreen"] * len(rowlbls),
        colColours =["palegreen"] * len(collbls),
        cellLoc ='center',
        colWidths=[0.05 for x in collbls],
        loc ='upper left')

    ax.set_title(ttl,fontweight ="bold")

    if(flnm != ''):
        plt.savefig(flnm+'.png',format='png',dpi=250)
    else:
        plt.show()
    plt.close()

    plt.show()
