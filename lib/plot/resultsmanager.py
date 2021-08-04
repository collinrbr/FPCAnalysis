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
