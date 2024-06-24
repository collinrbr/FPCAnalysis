# Wavelet Transform and the Wavelet Fourier Transform

The wavelet transform (WLT) measures the wavenumber and location of a wave (or equivalently frequency and time) of a 1D function, transforming it into a 2D function.

The WLT takes a specified wavelet and convolves it with input data. It is similar to a fourier transform with the added benefit that it can measure the location of a wave in a non-homogenous system.

There are two things to note about practically using the WLT.
1.) Since the WLT takes a 1D function and makes it into 2D, one can produce a very large data object by taking the WLT of a dense 1D array.
2.) Since the wavelet transfrom takes a 1D function and makes it into 2D, as there is not enough 'information' for the original 1D function to 'span' output 2D function, there is uncertainty between the wavenumber/position (or equivalently frequency/time). This uncertainty can be tuned by choice of the used wavelet. Here, we use the morlet wavelet and the parameter of interest is the 'simga' parameter. A value of 6 is standard here. Moreover, the morlet wavelet is curious as the uncertainty between the two coordinates changes as one increases wavenumber.

## Wavelet Fourier Transform

In this code, we use the wavelet fourier transform (WFT) which combines the the wavelet and fourier transforms. We fourier transform along any periodic dimensions and wavelet transfrom along dimensions that are not 'homogenous'.

