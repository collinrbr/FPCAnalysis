# debug.py>

# random plots typically related to previous debugging efforts

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_1d_dist(dparticles, parkey, vmax, x1, x2, y1, y2, flnm = ''):
    """
    Makes 1d distribution function of particle data within given spatial domain

    Parameters
    ----------
    dparticles : dict
        particle data dictionary
    parkey : str
        key of particle dictionary you want to plot (p1,p2,p3,x1,x2 or x3)
    x1 : float
        lower bound in xx space that you want to count
    x2 : float
        upper bound in xx space that you want to count
    y1 : float
        lower bound in xx space that you want to count
    y2 : float
        upper bound in xx space that you want to count
    flnm : str, optional
        specifies filename if plot is to be saved as png.
        if set to default, plt.show() will be called instead
    """

    #TODO: make gpts particles work for spatial domain
    gptsparticle = (x1 <= dparticles['x1'] ) & (dparticles['x1'] <= x2) & (y1 <= dparticles['x2'] ) & (dparticles['x2'] <= y2)
    print(gptsparticle)
    histdata = dparticles[parkey][gptsparticle]
    binsplt = np.linspace(-vmax,vmax,1000)

    plt.figure()
    plt.hist(histdata, bins = binsplt)
    plt.xlabel(parkey)
    plt.ylabel('n')

    if(flnm != ''):
        plt.savefig(flnm,format='png')
    else:
        plt.show()

    plt.close()

#TODO: figure out enerCEx and normalize it in compute energization
def check_JiEi_vs_CEi(dfields,dflow,dparticles,x,enerCEx,enerCEy,dx,xlim=None,ylim=None,zlim=None):
    """
    Comapres JiEi energization vs energization computed by integrating CEi over velocity space

    Parameters
    ----------
    dfields : dict
        field data dictionary from field_loader
    dflow : dict
        flow data dictionary from flow_loader
    dparticles : dict
        particle data dictionary
    x : 1d array
        1d coordinate data
    enerCex : 1d array
        energization by CEx from compute_energization function in analysis
    enerCey : 1d array
        energization by CEx from compute_energization function in analysis
    dx : float
        xx step size/ spacing
    xlim : [float,float]
        xx bounds of integration box used when normalizing energization by CEi
    ylim : [float,float]
        yy bounds of integration box used when normalizing energization by CEi
    zlim : [float,float]
        zz bounds of integration box used when normalizing energization by CEi
    """

    from lib.analysis import calc_Ji_Ei
    from lib.analysis import get_num_par_in_box

    #normalize by number of particles in box
    if xlim is not None:
        x1 = xlim[0]
        x2 = x1+dx
        xEnd = xlim[1]
    # If xlim is None, use lower x edge to upper x edge extents
    else:
        x1 = dfields['ex_xx'][0]
        x2 = x1 + dx
        xEnd = dfields['ex_xx'][-1]
    if ylim is not None:
        y1 = ylim[0]
        y2 = ylim[1]
    # If ylim is None, use lower y edge to lower y edge + dx extents
    else:
        y1 = dfields['ex_yy'][0]
        y2 = y1 + dx
    if zlim is not None:
        z1 = zlim[0]
        z2 = zlim[1]
    # If zlim is None, use lower z edge to lower z edge + dx extents
    else:
        z1 = dfields['ex_zz'][0]
        z2 = z1 + dx

    JxExplot = []
    JyEyplot = []
    JzEzplot = []
    enerCExplot = []
    enerCEyplot = []
    i = 0
    while(x2 <= xEnd):
        JxEx,JyEy,JzEz = calc_Ji_Ei(dfields,dflow,x1,x2,y1,y2,z1,z2)
        JxExplot.append(JxEx)
        JyEyplot.append(JyEy)
        JzEzplot.append(JzEz)

        numparinbox = get_num_par_in_box(dparticles,x1,x2,y1,y2,z1,z2)
        enerCExplot.append(enerCEx[i]/numparinbox)
        enerCEyplot.append(enerCEy[i]/numparinbox)
        x1 += dx
        x2 += dx
        i += 1

    plt.figure()
    plt.plot(x,JxExplot,label='JxEx')
    plt.plot(x,JyEyplot,label='JyEy')
    #plt.plot(x,JzEzplot,label='JzEz')
    plt.plot(x,enerCExplot)
    plt.plot(x,enerCEyplot)
    plt.xlabel('x (di)')
    plt.ylabel('Energy(x)')
    plt.legend()
    plt.show()

def plot_fluc_and_coords(B0,k,delB,epar,eperp1,eperp2,flnm = ''):

    font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16}

    xbasis = [1,0,0]
    ybasis = [0,1,0]
    zbasis = [0,0,1]
    veclabels = ['$\hat{x}$','$\hat{y}$','$\hat{z}$',r'$B_0$',r'$\hat{k}$',r'$\delta \hat{B}$',r'$\hat{e}_{||}$',r'$\hat{e}_{\perp 1}$',r'$\hat{e}_{\perp 2}$']

    B0norm = B0/np.linalg.norm(B0)
    knorm = k/np.linalg.norm(k)
    delBnorm = delB/np.linalg.norm(delB)

    #Build plot input varaibles
    U = [xbasis[0],ybasis[0],zbasis[0],B0norm[0],knorm[0],delBnorm[0],epar[0],eperp1[0],eperp2[0]]
    V = [xbasis[1],ybasis[1],zbasis[1],B0norm[1],knorm[1],delBnorm[1],epar[1],eperp1[1],eperp2[1]]
    W = [xbasis[2],ybasis[2],zbasis[2],B0norm[2],knorm[2],delBnorm[2],epar[2],eperp1[2],eperp2[2]]

    #all vectors start at origin
    X = []
    Y = []
    Z = []
    for i in range(0,len(U)):
        X.append(0)
        Y.append(0)
        Z.append(0)

    #TODO: consider coloring vectors
    fig, axs = plt.subplots(1, 3,figsize=(20,60))

    axs[0].set_xlim(-1.1,1.1)
    axs[0].set_ylim(-1.1,1.1)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].quiver(X, Y, U, V,scale=2.2) #scale is relative to plot limits
    for index, lbl in enumerate(veclabels):
        if(U[index] != 0 or V[index] != 0):
            axs[0].text(U[index], V[index], lbl, fontdict=font)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].grid()

    axs[1].set_xlim(-1.1,1.1)
    axs[1].set_ylim(-1.1,1.1)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].quiver(X, Z, U, W,scale=2.2) #scale is relative to plot limits
    for index, lbl in enumerate(veclabels):
        if(U[index] != 0 or W[index] != 0):
            axs[1].text(U[index], W[index], lbl, fontdict=font)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('z')
    axs[1].grid()

    axs[2].set_xlim(-1.1,1.1)
    axs[2].set_ylim(-1.1,1.1)
    axs[2].set_aspect('equal', adjustable='box')
    axs[2].quiver(Y, Z, V, W,scale=2.2) #scale is relative to plot limits
    for index, lbl in enumerate(veclabels):
        if(V[index] != 0 or W[index] != 0):
            axs[2].text(V[index], W[index], lbl, fontdict=font)
    axs[2].set_xlabel('y')
    axs[2].set_ylabel('z')
    axs[2].grid()

    plt.subplots_adjust(hspace=.5,wspace=.5)

    if(flnm != ''):
        plt.savefig(flnm,format='png')
    else:
        plt.show()

def plot_coords(epar,eperp1,eperp2,flnm = ''):

    font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16}

    xbasis = [1,0,0]
    ybasis = [0,1,0]
    zbasis = [0,0,1]
    veclabels = ['$\hat{x}$','$\hat{y}$','$\hat{z}$',r'$\hat{e}_{||}$',r'$\hat{e}_{\perp 1}$',r'$\hat{e}_{\perp 2}$']

    #Build plot input varaibles
    U = [xbasis[0],ybasis[0],zbasis[0],epar[0],eperp1[0],eperp2[0]]
    V = [xbasis[1],ybasis[1],zbasis[1],epar[1],eperp1[1],eperp2[1]]
    W = [xbasis[2],ybasis[2],zbasis[2],epar[2],eperp1[2],eperp2[2]]

    #all vectors start at origin
    X = []
    Y = []
    Z = []
    for i in range(0,len(U)):
        X.append(0)
        Y.append(0)
        Z.append(0)

    #TODO: consider coloring vectors
    fig, axs = plt.subplots(1, 3,figsize=(20,60))

    axs[0].set_xlim(-1.1,1.1)
    axs[0].set_ylim(-1.1,1.1)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].quiver(X, Y, U, V,scale=2.2) #scale is relative to plot limits
    for index, lbl in enumerate(veclabels):
        if(U[index] != 0 or V[index] != 0):
            axs[0].text(U[index], V[index], lbl, fontdict=font)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].grid()

    axs[1].set_xlim(-1.1,1.1)
    axs[1].set_ylim(-1.1,1.1)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].quiver(X, Z, U, W,scale=2.2) #scale is relative to plot limits
    for index, lbl in enumerate(veclabels):
        if(U[index] != 0 or W[index] != 0):
            axs[1].text(U[index], W[index], lbl, fontdict=font)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('z')
    axs[1].grid()

    axs[2].set_xlim(-1.1,1.1)
    axs[2].set_ylim(-1.1,1.1)
    axs[2].set_aspect('equal', adjustable='box')
    axs[2].quiver(Y, Z, V, W,scale=2.2) #scale is relative to plot limits
    for index, lbl in enumerate(veclabels):
        if(V[index] != 0 or W[index] != 0):
            axs[2].text(V[index], W[index], lbl, fontdict=font)
    axs[2].set_xlabel('y')
    axs[2].set_ylabel('z')
    axs[2].grid()

    plt.subplots_adjust(hspace=.5,wspace=.5)

    if(flnm != ''):
        plt.savefig(flnm,format='png')
    else:
        plt.show()

def plot_flucs_on_field_aligned_coords(dfields,k,delB,xlim,ylim,zlim,flnm = ''):

    from lib.analysis import compute_field_aligned_coord

    font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16}

    vperp2basis, vperp1basis, vparbasis = compute_field_aligned_coord(dfields,xlim,ylim,zlim)
    #make change of basis matrix
    _ = np.asarray([vparbasis,vperp1basis,vperp2basis]).T
    changebasismatrix = np.linalg.inv(_)

    kfldalg = np.matmul(changebasismatrix,k)
    delBfldalg = np.matmul(changebasismatrix,delB)

    epar = [1,0,0]
    eperp1 = [0,1,0]
    eperp2 = [0,0,1]
    veclabels = [r'$\hat{e}_{||}$',r'$\hat{e}_{\perp 1}$',r'$\hat{e}_{\perp 2}$',r'$k$',r'$\delta B$']


    #Build plot input varaibles
    U = [epar[0],eperp1[0],eperp2[0],kfldalg[0],np.real(delBfldalg[0])]
    V = [epar[1],eperp1[1],eperp2[1],kfldalg[1],np.real(delBfldalg[1])]
    W = [epar[2],eperp1[2],eperp2[2],kfldalg[2],np.real(delBfldalg[2])]

    pltlim = np.max([1,np.linalg.norm(kfldalg),np.linalg.norm(delBfldalg)])
    pltlim += .1*pltlim
    scale = 2*pltlim

    #all vectors start at origin
    X = []
    Y = []
    Z = []
    for i in range(0,len(U)):
        X.append(0)
        Y.append(0)
        Z.append(0)

    #TODO: consider coloring vectors
    fig, axs = plt.subplots(1, 3,figsize=(20,60))

    axs[0].set_xlim(-pltlim,pltlim)
    axs[0].set_ylim(-pltlim,pltlim)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].quiver(X, Y, U, V,scale=scale) #scale is relative to plot limits
    for index, lbl in enumerate(veclabels):
        if(U[index] != 0 or V[index] != 0):
            axs[0].text(U[index], V[index], lbl, fontdict=font)
    #axs[0].set_xlabel('x')
    #axs[0].set_ylabel('y')
    axs[0].grid()

    axs[1].set_xlim(-pltlim,pltlim)
    axs[1].set_ylim(-pltlim,pltlim)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].quiver(X, Z, U, W,scale=scale) #scale is relative to plot limits
    for index, lbl in enumerate(veclabels):
        if(U[index] != 0 or W[index] != 0):
            axs[1].text(U[index], W[index], lbl, fontdict=font)
    #axs[1].set_xlabel('x')
    #axs[1].set_ylabel('z')
    axs[1].grid()

    axs[2].set_xlim(-pltlim,pltlim)
    axs[2].set_ylim(-pltlim,pltlim)
    axs[2].set_aspect('equal', adjustable='box')
    axs[2].quiver(Y, Z, V, W,scale=scale) #scale is relative to plot limits
    for index, lbl in enumerate(veclabels):
        if(V[index] != 0 or W[index] != 0):
            axs[2].text(V[index], W[index], lbl, fontdict=font)
    #axs[2].set_xlabel('y')
    #axs[2].set_ylabel('z')
    axs[2].grid()

    plt.subplots_adjust(hspace=.5,wspace=.5)

    if(flnm != ''):
        plt.savefig(flnm,format='png')
    else:
        plt.show()
