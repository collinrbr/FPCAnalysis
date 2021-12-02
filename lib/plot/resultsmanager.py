# resultsmanager.py>

# functions related to making gifs and managing pngs/gifs

import numpy as np
import matplotlib.pyplot as plt
import os

def make_gif_from_folder(directory,flnm):
    #Not sure why this is necessary to break this up into a seperate function rather than including in make_superplot_gif
    #make gif
    import imageio #should import here as it might not be installed on every machine
    images = []
    filenames = os.listdir(directory)
    filenames = sorted(filenames)
    try:
        filenames.remove('.DS_store')
    except:
        pass

    print(filenames)

    for filename in filenames:
        images.append(imageio.imread(directory+'/'+filename))
    imageio.mimsave(flnm, images)

def keyname_to_plotname(keyname,axis):
    """
    Takes keyname and return string of appropriate format for plotting

    TODO: add flow keynames
    """
    plotname = ''

    if(keyname == 'ex'):
        plotname = '$E_x('
    elif(keyname == 'ey'):
        plotname = '$E_y('
    elif(keyname == 'ez'):
        plotname = '$E_z('
    elif(keyname == 'bx'):
        plotname = '$B_x('
    elif(keyname == 'by'):
        plotname = '$B_y('
    elif(keyname == 'bz'):
        plotname = '$B_z('

    if(axis== '_xx'):
        plotname = plotname + 'x)$'
    elif(axis == '_yy'):
        plotname = plotname + 'y)$'
    elif(axis == '_zz'):
        plotname = plotname + 'z)$'

    return plotname

def plume_keyname_to_plotname(keyname):
    """

    """
    plotname = ''

    if(keyname=='exr'):
        plotname = '$E_{x,r}$'
    elif(keyname=='exi'):
        plotname = '$E_{x,i}$'
    elif(keyname=='eyr'):
        plotname = '$E_{y,r}$'
    elif(keyname=='eyi'):
        plotname = '$E_{y,i}$'
    elif(keyname=='ezr'):
        plotname = '$E_{z,r}$'
    elif(keyname=='ezi'):
        plotname = '$E_{z,i}$'

    elif(keyname=='bxr'):
        plotname = '$B_{x,r}$'
    elif(keyname=='bxi'):
        plotname = '$B_{x,i}$'
    elif(keyname=='byr'):
        plotname = '$B_{y,r}$'
    elif(keyname=='byi'):
        plotname = '$B_{y,i}$'
    elif(keyname=='bzr'):
        plotname = '$B_{z,r}$'
    elif(keyname=='bzi'):
        plotname = '$B_{z,i}$'

    elif(keyname=='ux1r'):
        plotname = '$U_{x,r}$'
    elif(keyname=='ux1i'):
        plotname = '$U_{x,i}$'
    elif(keyname=='uy1r'):
        plotname = '$U_{y,r}$'
    elif(keyname=='uy1i'):
        plotname = '$U_{y,i}$'
    elif(keyname=='uz1r'):
        plotname = '$U_{z,r}$'
    elif(keyname=='uz1i'):
        plotname = '$U_{z,i}$'

    elif(keyname=='kpar'):
        plotname = '$k_{||} d_i$'
    elif(keyname=='kperp'):
        plotname = '$k_{\perp} d_i$'

    return plotname
