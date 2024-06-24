# Instability Isolation Method

The Instability Isolate Method (IIM) from Brown et al 2023 [https://doi.org/10.1017/S0022377823000478](https://doi.org/10.1017/S0022377823000478), uses a combination of the wavelet transform and faradays law, assuming plane wave solutions, to locally compute the measured dispersion relation (kx,ky,kz,omega) within some singular element of a linear superposition of the total electromagnetic fields. Given a specified (kx,ky,kz) and position (x) the IIM will return the (omega) associated with the system at that location. Then, one can compare the measured valued to the expected value for different modes, verifying that the particular element of the linear superposition contains the desired physics in isolation.

In shock systems, we assume that the waves produced by instabilities in the system are linearly separable from the 'main shock profile'. We then can compute the dispersion relation for just the instability fields term and compare it to linear dispersion relates to show that our separated term contains the desired physics.

One should note that the 'isolation' part of the IIM occurs in the first step, when one chooses a linear superposition of the total electromagnetic fields. The remainder of the steps is to self verify that the term contains the desired wave physics, and thus that the IIM method isolates the wave produced by the instability.

Often, we use the IIM to show that an isolation (aka linear superposition) is meaningful before computed the FPC with just that term.
