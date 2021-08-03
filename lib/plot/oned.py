# 1dfields.py>

# functions related to plotting 1d field data

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_field(dfields, fieldkey, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0, axvx1 = float('nan'), axvx2 = float('nan'), flnm = ''):
    """
    Plots field data at some static frame down a line along x,y,z for some
    selected field.

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    axis : str, optional
        name of axis you want to plot along (_xx, _yy, _zz)
    xxindex : int, optional
        index of data along xx axis (ignored if axis = '_xx')
    yyindex : int, optional
        index of data along yy axis (ignored if axis = '_yy')
    zzindex : int, optional
        index of data along zz axis (ignored if axis = '_zz')
    axvx1 : float, optional
        x position of vertical line on plot
    axvx2 : float, optional
        x position of vertical line on plot
    """


    if(axis == '_zz'):
        fieldval = np.asarray([dfields[fieldkey][i][yyindex][xxindex] for i in range(0,len(dfields[fieldkey+axis]))])
        xlbl = 'z'
    elif(axis == '_yy'):
        fieldval = np.asarray([dfields[fieldkey][zzindex][i][xxindex] for i in range(0,len(dfields[fieldkey+axis]))])
        xlbl = 'y'
    elif(axis == '_xx'):
        fieldval = np.asarray([dfields[fieldkey][zzindex][yyindex][i] for i in range(0,len(dfields[fieldkey+axis]))])
        xlbl = 'x'

    fieldcoord = np.asarray(dfields[fieldkey+axis])

    plt.figure(figsize=(20,10))
    plt.xlabel(xlbl)
    plt.ylabel(fieldkey)
    plt.plot(fieldcoord,fieldval)
    plt.grid()
    if(not(axvx1 != axvx1)): #if not nan
        plt.axvline(x=axvx1)
    if(not(axvx2 != axvx2)): #if not nan
        plt.axvline(x=axvx2)
    if(flnm == ''):
        plt.show()
    else:
        plt.savefig(flnm,format='png')
    plt.close()

def plot_all_fields(dfields, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0, flnm = '',lowxlim=None, highxlim=None):
    """
    Plots all field data at some static frame down a line along x,y,z.

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    axis : str, optional
        name of axis you want to plot along (_xx, _yy, _zz)
    xxindex : int, optional
        index of data along xx axis (ignored if axis = '_xx')
    yyindex : int, optional
        index of data along yy axis (ignored if axis = '_yy')
    zzindex : int, optional
        index of data along zz axis (ignored if axis = '_zz')
    """
    if(axis == '_zz'):
        ex = np.asarray([dfields['ex'][i][yyindex][xxindex] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][i][yyindex][xxindex] for i in range(0,len(dfields['ey'+axis]))])
        ez = np.asarray([dfields['ez'][i][yyindex][xxindex] for i in range(0,len(dfields['ez'+axis]))])
        bx = np.asarray([dfields['bx'][i][yyindex][xxindex] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][i][yyindex][xxindex] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][i][yyindex][xxindex] for i in range(0,len(dfields['bz'+axis]))])
    elif(axis == '_yy'):
        ex = np.asarray([dfields['ex'][zzindex][i][xxindex] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][zzindex][i][xxindex] for i in range(0,len(dfields['ex'+axis]))])
        ez = np.asarray([dfields['ez'][zzindex][i][xxindex] for i in range(0,len(dfields['ex'+axis]))])
        bx = np.asarray([dfields['bx'][zzindex][i][xxindex] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][zzindex][i][xxindex] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][zzindex][i][xxindex] for i in range(0,len(dfields['bz'+axis]))])
    elif(axis == '_xx'):
        ex = np.asarray([dfields['ex'][zzindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        ey = np.asarray([dfields['ey'][zzindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        ez = np.asarray([dfields['ez'][zzindex][yyindex][i] for i in range(0,len(dfields['ex'+axis]))])
        bx = np.asarray([dfields['bx'][zzindex][yyindex][i] for i in range(0,len(dfields['bx'+axis]))])
        by = np.asarray([dfields['by'][zzindex][yyindex][i] for i in range(0,len(dfields['by'+axis]))])
        bz = np.asarray([dfields['bz'][zzindex][yyindex][i] for i in range(0,len(dfields['bz'+axis]))])

    fieldcoord = np.asarray(dfields['ex'+axis]) #assumes all fields have same coordinates

    fig, axs = plt.subplots(6,figsize=(20,10))
    axs[0].plot(fieldcoord,ex,label="ex")
    axs[0].set_ylabel("$ex$")
    axs[0].grid()
    if(lowxlim != None and highxlim != None):
        axs[0].set_xlim(lowxlim,highxlim)
    axs[1].plot(fieldcoord,ey,label='ey')
    axs[1].set_ylabel("$ey$")
    axs[1].grid()
    if(lowxlim is not None and highxlim is not None):
        axs[1].set_xlim(lowxlim,highxlim)
    axs[2].plot(fieldcoord,ez,label='ez')
    axs[2].set_ylabel("$ez$")
    axs[2].grid()
    if(lowxlim is not None and highxlim is not None):
        axs[2].set_xlim(lowxlim,highxlim)
    axs[3].plot(fieldcoord,bx,label='bx')
    axs[3].set_ylabel("$bx$")
    axs[3].grid()
    if(lowxlim is not None and highxlim is not None):
        axs[3].set_xlim(lowxlim,highxlim)
    axs[4].plot(fieldcoord,by,label='by')
    axs[4].set_ylabel("$by$")
    axs[4].grid()
    if(lowxlim is not None and highxlim is not None):
        axs[4].set_xlim(lowxlim,highxlim)
    axs[5].plot(fieldcoord,bz,label='bz')
    axs[5].set_ylabel("$bz$")
    axs[5].grid()
    if(lowxlim is not None and highxlim is not None):
        axs[5].set_xlim(lowxlim,highxlim)
    if(axis == '_xx'):
        axs[5].set_xlabel("$x$")
    elif(axis == '_yy'):
        axs[5].set_xlabel("$y$")
    elif(axis == '_yy'):
        axs[5].set_xlabel("$z$")
    plt.subplots_adjust(hspace=0.5)
    if(flnm == ''):
        plt.show()
    else:
        plt.savefig(flnm,format='png')
    plt.close()


def plot_flow(dflow, flowkey, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0, axvx1 = float('nan'), axvx2 = float('nan'), flnm = ''):
    """
    Plots flow data

    Parameters
    ----------
    dflow : dict
        flow data dictionary from flow_loader
    flowkey : str
        name of flow you want to plot (ux, uy, uz)
    xxindex : int, optional
        index of data along xx axis (ignored if axis = '_xx')
    yyindex : int, optional
        index of data along yy axis (ignored if axis = '_yy')
    zzindex : int, optional
        index of data along zz axis (ignored if axis = '_zz')
    axvx1 : float, optional
        x position of vertical line on plot
    axvx2 : float, optional
        x position of vertical line on plot
    """
    if(axis == '_zz'):
        flowval = np.asarray([dflow[flowkey][i][yyindex][xxindex] for i in range(0,len(dflow[flowkey+axis]))])
        xlbl = 'z'
    elif(axis == '_yy'):
        flowval = np.asarray([dflow[flowkey][zzindex][i][xxindex] for i in range(0,len(dflow[flowkey+axis]))])
        xlbl = 'y'
    elif(axis == '_xx'):
        flowval = np.asarray([dflow[flowkey][zzindex][yyindex][i] for i in range(0,len(dflow[flowkey+axis]))])
        xlbl = 'x'

    flowcoord = np.asarray(dflow[flowkey+axis])

    plt.figure(figsize=(20,10))
    plt.xlabel(xlbl)
    plt.ylabel(flowkey)
    plt.grid()
    if(not(axvx1 != axvx1)): #if not nan
        plt.axvline(x=axvx1)
    if(not(axvx2 != axvx2)): #if not nan
        plt.axvline(x=axvx2)
    plt.plot(flowcoord,flowval)
    if(flnm == ''):
        plt.show()
    else:
        plt.savefig(flnm,format='png')
    plt.close()

def plot_all_flow(dflow, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0, flnm = ''):
    """
    Plots all flow data at some static frame down a line along x,y,z.

    Parameters
    ----------
    dflow : dict
        flow data dictionary from flow_loader
    axis : str, optional
        name of axis you want to plot along (_xx, _yy, _zz)
    xxindex : int, optional
        index of data along xx axis (ignored if axis = '_xx')
    yyindex : int, optional
        index of data along yy axis (ignored if axis = '_yy')
    zzindex : int, optional
        index of data along zz axis (ignored if axis = '_zz')
    """

    if(axis == '_zz'):
        ux = np.asarray([dflow['ux'][i][yyindex][xxindex] for i in range(0,len(dflow['ux'+axis]))])
        uy = np.asarray([dflow['uy'][i][yyindex][xxindex] for i in range(0,len(dflow['uy'+axis]))])
        uz = np.asarray([dflow['uz'][i][yyindex][xxindex] for i in range(0,len(dflow['uz'+axis]))])
    elif(axis == '_yy'):
        ux = np.asarray([dflow['ux'][zzindex][i][xxindex] for i in range(0,len(dflow['ux'+axis]))])
        uy = np.asarray([dflow['uy'][zzindex][i][xxindex] for i in range(0,len(dflow['uy'+axis]))])
        uz = np.asarray([dflow['uz'][zzindex][i][xxindex] for i in range(0,len(dflow['uz'+axis]))])
    elif(axis == '_xx'):
        ux = np.asarray([dflow['ux'][zzindex][yyindex][i] for i in range(0,len(dflow['ux'+axis]))])
        uy = np.asarray([dflow['uy'][zzindex][yyindex][i] for i in range(0,len(dflow['uy'+axis]))])
        uz = np.asarray([dflow['uz'][zzindex][yyindex][i] for i in range(0,len(dflow['uz'+axis]))])

    fieldcoord = np.asarray(dflow['ux'+axis]) #assumes all fields have same coordinates

    fig, axs = plt.subplots(3,figsize=(20,10))
    axs[0].plot(fieldcoord,ux,label="vx")
    axs[0].set_ylabel("$ux$")
    axs[0].grid()
    axs[1].plot(fieldcoord,uy,label='vy')
    axs[1].set_ylabel("$uy$")
    axs[1].grid()
    axs[2].plot(fieldcoord,uz,label='vz')
    axs[2].set_ylabel("$uz$")
    axs[2].grid()
    if(axis == '_xx'):
        axs[2].set_xlabel("$x$")
    elif(axis == '_yy'):
        axs[2].set_xlabel("$y$")
    elif(axis == '_yy'):
        axs[2].set_xlabel("$z$")
    plt.subplots_adjust(hspace=0.5)
    if(flnm != ''):
        plt.savefig(flnm,format='png')
    else:
        plt.show()

def plot_field_time(dfieldsdict, fieldkey, xxindex = 0, yyindex = 0, zzindex = 0):
    """
    Plots field at static location as a function of time.

    Parameters
    ----------
    dfieldsdict : dict
        dictonary of dfields and corresponding frame number from all_field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    xxindex : int
        index of data along xx axis
    yyindex : int
        index of data along yy axis
    zzindex : int
        index of data along zz axis
    """

    fieldval = np.asarray([dfields[fieldkey][zzindex][yyindex][xxindex] for dfields in dfieldsdict['dfields']])
    xlbl = 't'

    fieldcoord = np.asarray(dfieldsdict['frame'])

    plt.figure(figsize=(20,10))
    plt.xlabel(xlbl)
    plt.ylabel(fieldkey)
    plt.plot(fieldcoord,fieldval)
    plt.show()

def time_stack_line_plot(dfieldsdict, fieldkey, pts = [], axis = '_xx', xxindex = 0, yyindex = 0, zzindex = 0):
    """
    Plots field data at some static frame down a line along x,y,z for some
    selected field for each frame in seperate panels

    This plot is primarily used to test shock_from_ex_cross

    Parameters
    ----------
    dfieldsdict : dict
        dictonary of dfields and corresponding frame number from all_field_loader
    fieldkey : str
        name of field you want to plot (ex, ey, ez, bx, by, bz)
    pts : 1d array
        array containing independent axis position of singular point to be plotted on each panel
        point will have dependent axis position equal to the selected field
    axis : str, optional
        name of axis you want to plot along (_xx, _yy, _zz)
    xxindex : int, optional
        index of data along xx axis (ignored if axis = '_xx')
    yyindex : int, optional
        index of data along yy axis (ignored if axis = '_yy')
    zzindex : int, optional
        index of data along zz axis (ignored if axis = '_zz')
    """

    fig, axs = plt.subplots(len(dfieldsdict['frame']), sharex=True, sharey=True)
    fieldcoord = np.asarray(dfieldsdict['dfields'][0][fieldkey+axis])
    fig.set_size_inches(18.5, 30.)

    #sbpltlocation = len(dfielddict['frame'])+10+1
    for k in range(0,len(dfieldsdict['frame'])):

        #_ax = plt.subplots(len(dfielddict['frame']),k,sharex=True)
        if(axis == '_zz'):
            fieldval = np.asarray([dfieldsdict['dfields'][k][fieldkey][i][yyindex][xxindex] for i in range(0,len(dfieldsdict['dfields'][k][fieldkey+axis]))])
            xlbl = 'z'
        elif(axis == '_yy'):
            fieldval = np.asarray([dfieldsdict['dfields'][k][fieldkey][zzindex][i][xxindex] for i in range(0,len(dfieldsdict['dfields'][k][fieldkey+axis]))])
            xlbl = 'y'
        elif(axis == '_xx'):
            fieldval = np.asarray([dfieldsdict['dfields'][k][fieldkey][xxindex][yyindex][i] for i in range(0,len(dfieldsdict['dfields'][k][fieldkey+axis]))])
            xlbl = 'x'

        axs[k].plot(fieldcoord,fieldval)
        if(len(pts) > 0):
            axs[k].scatter([pts[k]],[0.])
        #axs[k].ylabel(fieldkey+'(frame = '+str(dfielddict['frame'][k])+')')

    plt.show()

def plot_stack_field_along_x(dfields,fieldkey,stackaxis,yyindex=0,zzindex=0,xlow=None,xhigh=None):
    """

    """
    if(stackaxis != '_yy' and stackaxis != '_zz'):
        print("Please stack along _yy or _zz")

    plt.figure()
    fieldcoord = np.asarray(dfields[fieldkey+'_xx'])
    for k in range(0,len(dfields[fieldkey+stackaxis])):
        fieldval = np.asarray([dfields[fieldkey][zzindex][yyindex][i] for i in range(0,len(dfields[fieldkey+'_xx']))])
        if(stackaxis == '_yy'):
            yyindex += 1
        elif(stackaxis == '_zz'):
            zzindex += 1
        plt.xlabel('x')
        plt.ylabel(fieldkey)
        if(xlow != None and xhigh != None):
            plt.xlim(xlow,xhigh)
        plt.plot(fieldcoord,fieldval)
    plt.grid()
    plt.show()
    plt.close()

def plot_compression_ratio(dfields, upstreambound, downstreambound, xxindex=0, yyindex=0, zzindex=0, flnm=''):
    """
    Plots Bz(x), along with vertical lines at the provided upstream and downstream
    bounds.

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    downstreambound : float
        x position of the end of the upstream position
    upstreambound : float
        x position of the end of the downstream position
    """

    from lib.frametransform import get_comp_ratio

    ratio,bzup,bzdown = get_comp_ratio(dfields,upstreambound,downstreambound)

    fieldkey = 'bz'
    axis='_xx'

    plt.figure(figsize=(20,10))
    fieldval = np.asarray([dfields[fieldkey][zzindex][yyindex][i] for i in range(0,len(dfields[fieldkey+axis]))])
    xlbl = 'x'

    fieldcoord = np.asarray(dfields[fieldkey+axis])

    plt.figure(figsize=(20,10))
    plt.xlabel(xlbl)
    plt.ylabel(fieldkey)
    plt.plot(fieldcoord,fieldval)
    plt.axvline(x=upstreambound)
    plt.axvline(x=downstreambound)
    plt.plot([dfields[fieldkey+axis][0],downstreambound],[bzdown,bzdown]) #line showing average bz downstream
    plt.plot([upstreambound,dfields[fieldkey+axis][-1]],[bzup,bzup]) #line showing average bz upstream
    if(flnm == ''):
        plt.show()
    else:
        plt.savefig(flnm+'.png',format='png')
    plt.close()

def compare_fields(dfields1, dfields2, fieldkey, axis='_xx', xxindex = 0, yyindex = 0, zzindex = 0, axvx1 = float('nan'), axvx2 = float('nan'), flnm = ''):
    """

    """


    if(axis == '_zz'):
        fieldval1 = np.asarray([dfields1[fieldkey][i][yyindex][xxindex] for i in range(0,len(dfields1[fieldkey+axis]))])
        fieldval2 = np.asarray([dfields2[fieldkey][i][yyindex][xxindex] for i in range(0,len(dfields2[fieldkey+axis]))])
        xlbl = 'z'
    elif(axis == '_yy'):
        fieldval1 = np.asarray([dfields1[fieldkey][zzindex][i][xxindex] for i in range(0,len(dfields1[fieldkey+axis]))])
        fieldval2 = np.asarray([dfields2[fieldkey][zzindex][i][xxindex] for i in range(0,len(dfields2[fieldkey+axis]))])
        xlbl = 'y'
    elif(axis == '_xx'):
        fieldval1 = np.asarray([dfields1[fieldkey][zzindex][yyindex][i] for i in range(0,len(dfields1[fieldkey+axis]))])
        fieldval2 = np.asarray([dfields2[fieldkey][zzindex][yyindex][i] for i in range(0,len(dfields2[fieldkey+axis]))])
        xlbl = 'x'

    fieldcoord1 = np.asarray(dfields1[fieldkey+axis])
    fieldcoord2 = np.asarray(dfields2[fieldkey+axis])

    plt.figure(figsize=(20,10))
    plt.xlabel(xlbl)
    plt.ylabel(fieldkey)
    plt.plot(fieldcoord1,fieldval1)
    plt.plot(fieldcoord2,fieldval2)
    if(not(axvx1 != axvx1)): #if not nan
        plt.axvline(x=axvx1)
    if(not(axvx2 != axvx2)): #if not nan
        plt.axvline(x=axvx2)
    if(flnm == ''):
        plt.show()
    else:
        plt.savefig(flnm,format='png')
    plt.close()

def make_field_scan_gif(dfields, fieldkey, directory, axis='_xx'):

    try:
        os.mkdir(directory)
    except:
        pass

    sweepvar = dfields[fieldkey+'_xx'][:]

    for i in range(0,len(sweepvar)):
        print('Making plot '+str(i)+' of '+str(len(sweepvar)))
        flnm = directory+'/'+str(i).zfill(6)
        plot_field(dfields, fieldkey, axis='_xx', xxindex = i, yyindex = 0, zzindex = 0, axvx1 = dfields[fieldkey+'_xx'][i], axvx2 = float('nan'), flnm = flnm)
