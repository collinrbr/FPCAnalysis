import numpy as np
import math

def kaw_curve(kperp,kpar,beta_i,tau,
              comp_error_prop=False,delta_kperp = 0., delta_kpar = 0., delta_beta_i = 0., delta_tau = 0.):
    """
    Empirical dispersion relation for a kinetic alfven wave

    From Howes et al. 2014

    Parameters
    ----------
    kperp : float
        kperp value
    kpar : float
        kpar value
    beta_i : float
        ion plasma beta
    tau : float
        Ti/Te temperature ratio
    comp_error_prop : bool, opt
        if true, propogates error
    delta_kperp,delta_kpar,delta_beta_i,delta_tau : float
        source error

    Returns
    -------
    omega_over_Omega_i : float
        ratio of freq to gyrofreq
    """

    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        kperp = ufloat(kperp,delta_kperp)
        kpar = ufloat(kpar,delta_kpar)
        beta_i = ufloat(beta_i,delta_beta_i)
        tau = ufloat(tau,delta_tau)

        omega_over_Omega_i=kpar*sqrt(1.+ kperp**2./(1.+ 2./(beta_i*(1.+1./tau))))

    else:
        omega_over_Omega_i=kpar*math.sqrt(1.+ kperp**2./(1.+ 2./(beta_i*(1.+1./tau))))

    return omega_over_Omega_i

def fastmagson_curve(kperp,kpar,beta_i,tau,
              comp_error_prop=False,delta_kperp = 0., delta_kpar = 0., delta_beta_i = 0., delta_tau = 0.):
    """
    Dispersion for fast magnetosonic wave

    From Klein et al. 2012

    Parameters
    ----------
    kperp : float
        kperp value
    kpar : float
        kpar value
    beta_i : float
        ion plasma beta
    tau : float
        Ti/Te temperature ratio
    comp_error_prop : bool, opt
        if true, propogates error
    delta_kperp,delta_kpar,delta_beta_i,delta_tau : float
        source error

    Returns
    -------
    omega_over_Omega_i : float
        ratio of freq to gyrofreq
    """

    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        kperp = ufloat(kperp,delta_kperp)
        kpar = ufloat(kpar,delta_kpar)
        beta_i = ufloat(beta_i,delta_beta_i)
        tau = ufloat(tau,delta_tau)

        bt = beta_i*(1+1./tau)
        k_tot = sqrt(kperp**2.+kpar**2.)

        omega_over_Omega_i=k_tot*sqrt( (1.+bt + sqrt( (1.+bt)**2. -4.*bt*(kpar/k_tot)**2.))/2.)

    else:
        bt = beta_i*(1+1./tau)
        k_tot = math.sqrt(kperp**2.+kpar**2.)

        omega_over_Omega_i=k_tot*math.sqrt( (1.+bt + math.sqrt( (1.+bt)**2. -4.*bt*(kpar/k_tot)**2.))/2.)

    return omega_over_Omega_i

def slowmagson_curve(kperp,kpar,beta_i,tau,
              comp_error_prop=False,delta_kperp = 0., delta_kpar = 0., delta_beta_i = 0., delta_tau = 0.):
    """
    Dispersion for slow magnetosonic wave

    From Klein et al. 2012

    Parameters
    ----------
    kperp : float
        kperp value
    kpar : float
        kpar value
    beta_i : float
        ion plasma beta
    tau : float
        Ti/Te temperature ratio
    comp_error_prop : bool, opt
        if true, propogates error
    delta_kperp,delta_kpar,delta_beta_i,delta_tau : float
        source error

    Returns
    -------
    omega_over_Omega_i : float
        ratio of freq to gyrofreq
    """
    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        kperp = ufloat(kperp,delta_kperp)
        kpar = ufloat(kpar,delta_kpar)
        beta_i = ufloat(beta_i,delta_beta_i)
        tau = ufloat(tau,delta_tau)

        bt = beta_i*(1+1./tau)
        k_tot = sqrt(kperp**2.+kpar**2.)

        omega_over_Omega_i=k_tot*sqrt( (1.+bt - sqrt( (1.+bt)**2. -4.*bt*(kpar/k_tot)**2.))/2.)

    else:
        bt = beta_i*(1+1./tau)
        k_tot = math.sqrt(kperp**2.+kpar**2.)

        omega_over_Omega_i=k_tot*math.sqrt( (1.+bt - math.sqrt( (1.+bt)**2. -4.*bt*(kpar/k_tot)**2.))/2.)

    return omega_over_Omega_i

def whistler_curve(kperp,kpar,beta_i,tau,
              comp_error_prop=False,delta_kperp = 0., delta_kpar = 0., delta_beta_i = 0., delta_tau = 0.):
    """
    Analytical limit of fastmagson_curve(kperp,kpar) when kpar << kperp

    Parameters
    ----------
    kperp : float
        kperp value
    kpar : float
        kpar value
    beta_i : float
        ion plasma beta
    tau : float
        Ti/Te temperature ratio
    comp_error_prop : bool, opt
        if true, propogates error
    delta_kperp,delta_kpar,delta_beta_i,delta_tau : float
        source error

    Returns
    -------
    omega_over_Omega_i : float
        ratio of freq to gyrofreq
    """
    if(comp_error_prop):
        from uncertainties import ufloat
        from uncertainties.umath import sqrt

        kperp = ufloat(kperp,delta_kperp)
        kpar = ufloat(kpar,delta_kpar)
        beta_i = ufloat(beta_i,delta_beta_i)
        tau = ufloat(tau,delta_tau)

        k_tot = sqrt(kperp**2.+kpar**2.)
        omega_over_Omega_i=k_tot*sqrt(1.+beta_i*(1.+1./tau) + kpar**2.)
    else:
        k_tot = math.sqrt(kperp**2.+kpar**2.)
        omega_over_Omega_i=k_tot*math.sqrt(1.+beta_i*(1.+1./tau) + kpar**2.)

    return omega_over_Omega_i

