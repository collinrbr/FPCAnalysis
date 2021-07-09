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
