"""

Created on April 2020

@author: Francis Santos

"""
import rampy as rp
import numpy as np
from scipy.signal import find_peaks, peak_widths, medfilt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import mode
from ..utils import add_doc

def _1gaussian(x, intensity, center, width):
    return intensity * np.exp(-(x-center)**2 / width)


def _2gaussian(x, intensity1, center1, width1, intensity2, center2, width2):
    return intensity1 * np.exp(-(x-center1)**2 / width1) + \
            intensity2 * np.exp(-(x-center2)**2 / width2)


def _1lorentzian(x, intensity, center, width):
    return intensity * 1/(1 + ((x-center) / width)**2)


def _2lorentzian(x, intensity1, center1, width1, intensity2, center2, width2):
    return intensity1 * 1/(1 + ((x-center1) / width1)**2) + \
            intensity2 * 1/(1 + ((x-center2) / width2)**2)


def _1voigtian(x, intensity, center, width, lor_gaus):
    return intensity * (lor_gaus * (_1gaussian(x, intensity, center, width)/intensity) + (1-lor_gaus) * (_1lorentzian(x, intensity, center, width)/intensity))


def _1pearsonIV(x, intensity, position, width, shape_param1, shape_param2):
    return intensity * (1 + ((x - position) / width)**2)**(-shape_param1) * (np.exp(-shape_param2 * np.arctan((x - position) / width)))


@add_doc(
    '''
        Method to remove the baseline using RamPy's baseline function.


        Parameters
        ----------
        spectra: np.ndarray
            Spectroscopy data to process.
        roi: np.ndarray
            Region of interest to pay attention and keep during the application of the algorithm
        method: str
            Method used for baseline removal:
                "poly": polynomial fitting, with splinesmooth the degree of the polynomial.
                "unispline": spline with the UnivariateSpline function of Scipy, splinesmooth is
                            the spline smoothing factor (assume equal weight in the present case);
                "gcvspline": spline with the gcvspl.f algorythm, really robust.
                            Spectra must have x, y, ese in it, and splinesmooth is the smoothing factor;
                            For gcvspline, if ese are not provided we assume ese = sqrt(y).
                            WARNING: Requires the installation of the gcvspline Python package prior to use in the Python ENV used by Julia.
                            See website for install instructions
                "exp": exponential background;
                "log": logarythmic background;
                "rubberband": rubberband baseline fitting;
                "als": automatic least square fitting following Eilers and Boelens 2005;
                "arPLS": automatic baseline fit using the algorithm from Baek et al. 2015
                "drpPLS" (DEFAULT): Baseline correction using asymmetrically reweighted penalized least squares smoothing, Analyst 140: 250-257.
        index: (int, tuple, list, np.ndarray)
            Index/indices to preprocess.
        inplace: bool
            Perform the change in the Spectra object (True) or create a new one and return (False)

            kwargs
            ------
            polynomial_order : Int
                The degree of the polynomial (0 for a constant), default = 1.
            s : Float
                spline smoothing coefficient for the unispline and gcvspline algorithms.
            lam : Float
                float, the lambda smoothness parameter for the ALS, ArPLS and drPLS algorithms. Typical values are between 10**2 to 10**9, default = 10**5 for ALS and ArPLS and default = 10**6 for drPLS.
            p : Float
                float, for the ALS algorithm, advised value between 0.001 to 0.1, default = 0.01.
            ratio : float
                ratio parameter of the arPLS and drPLS algorithm. default = 0.01 for arPLS and 0.001 for drPLS.
            niter : Int
                number of iteration of the ALS and drPLS algorithm, default = 10 for ALS and default = 100 for drPLS.
            eta : Float
                roughness parameter for the drPLS algorithm, is between 0 and 1, default = 0.5
            p0_exp : List
                containg the starting parameter for the exp baseline fit with curve_fit. Default = [1.,1.,1.].
            p0_log : List
                containg the starting parameter for the log baseline fit with curve_fit. Default = [1.,1.,1.,1.].
        
        
        Returns
        -------
        if(inplace):
            None
        else:
            New Spectra object with preprocessed signal


    '''
)
def _removeBaseline(spectra, roi, method, index=-1, inplace=False, **kwargs):
    if inplace:
        spectra_c = spectra
    else:
        spectra_c = spectra.copy()

    if(index == -1):  # All signals
        for i in spectra_c.index:
            new_sig, __ = rp.baseline(spectra_c.wavenumbers,
                                      spectra_c.intensity[i], roi,
                                      method, **kwargs)
            spectra_c.intensity[i] = new_sig.reshape(-1,)
    else:
        if(isinstance(index, (tuple, list, np.ndarray))):  # Multiple signals
            for i in index:
                new_sig, __ = rp.baseline(spectra_c.wavenumbers,
                                          spectra_c.intensity[i], roi,
                                          method, **kwargs)
                spectra_c.intensity[i] = new_sig.reshape(-1,)
        elif(isinstance(index, int)):  # Only 1 signal
            new_sig, __ = rp.baseline(spectra_c.wavenumbers,
                                      spectra_c.intensity[index], roi,
                                      method, **kwargs)
            spectra_c.intensity[index] = new_sig.reshape(-1,)

    if not inplace:
        return spectra_c


@add_doc(
    '''
        Method to smooth the signal using the RamPy library.

        
        Paremeters
        ----------
        spectra: np.ndarray
        index: (int, tuple, array, np.ndarray)
            Index/indices to preprocess.
        method: str
            Method for smoothing the signal;
            choose between savgol (Savitzky-Golay),
            GCVSmoothedNSpline, MSESmoothedNSpline,
            DOFSmoothedNSpline,
            whittaker,
            'flat',
            'hanning',
            'hamming',
            'bartlett',
            'blackman'.
        inplace: bool            
            Perform the change in the Spectra object (True) or create a new one and return (False)
        

        kwargs
        ------
        window_length : int
            The length of the filter window (i.e. the number of coefficients). window_length must be a positive odd integer.
        polyorder : int
            The order of the polynomial used to fit the samples. polyorder must be less than window_length.
        Lambda : float
            smoothing parameter of the Whittaker filter described in Eilers (2003). The higher the smoother the fit.
        d : int
            d parameter in Whittaker filter, see Eilers (2003).
        ese_y : ndarray
            errors associated with y (for the gcvspline algorithms)

        Returns
        -------
        if(inplace):
            None
        else:
            New Spectra object with preprocessed signal

    '''
)
def _smoothSignal(spectra, index=-1, method="flat", inplace=False,
                  **kwargs):
    if inplace:
        spectra_c = spectra
    else:
        spectra_c = spectra.copy()

    if(index == -1):  # All signals
        for i in spectra_c.index:
            new_sig = rp.smooth(spectra_c.wavenumbers,
                                spectra_c.intensity[i],
                                method, **kwargs)
            spectra_c.intensity[i] = new_sig[:len(spectra_c.wavenumbers)]
    else:
        if(isinstance(index, tuple)):  # Multiple signals
            for i in index:
                new_sig = rp.smooth(spectra_c.wavenumbers,
                                    spectra_c.intensity[i],
                                    method, **kwargs)
                spectra_c.intensity[i] = new_sig[:len(spectra_c.wavenumbers)]
        elif(isinstance(index, int)):  # Only 1 signal
            new_sig = rp.smooth(spectra_c.wavenumbers,
                                spectra_c.intensity[index],
                                method, **kwargs)
            spectra_c.intensity[index] = new_sig[:len(spectra_c.wavenumbers)]

    if not inplace:
        return spectra_c


@add_doc(
    '''

    Detect the peaks using function fitting to Gaussian and Lorentzian functions
    and (optionally) plot them.


    Parameters
    ----------
    spectra: np.ndarray
        Spectroscopy data of interest.
    index: int
        Spectrum number sample of interest.
    do_plot: bool
        If True plots the peaks detected.


    Returns
    -------
    None

    '''
)
def _detectPeaks(spectra, index=0, do_plot=True):
    # Obtain the full signals in a way they can be accessed more easily later
    x = spectra.wavenumbers
    y = spectra.intensity[index]

    # Find peaks and information related to it (width, floor, start and end)
    peaks, __ = find_peaks(y, prominence=y.mean()/(x.size/5000))
    peak_widths_info = peak_widths(y, peaks, rel_height=0.5)
    widths = peak_widths_info[0]
    widths_pos_init = peak_widths_info[2]
    widths_pos_end = peak_widths_info[3]
    if(do_plot):
        plt.figure(figsize=(15, 8))
        plt.plot(x, y, linewidth=2)
        plt.xlim([x.min(), x.max()])

    # For every peak with these functions
    funcs = (_1gaussian, _1lorentzian)
    for i in range(0, len(peaks)):
        results = {}
        peak = peaks[i]
        initial_fitting = np.array([y[peak], x[peak], widths[i]])

        if x[int(widths_pos_init[i]):int(widths_pos_end[i])].size < 3: # Checking that the peak is not just two adjacent points
            continue
        try:
            for func in funcs:
                result = curve_fit(func, x[int(widths_pos_init[i]):int(widths_pos_end[i])], y[int(widths_pos_init[i]):int(widths_pos_end[i])], p0=initial_fitting)
                results[func] = result
        except RuntimeError:
            pass

        if(len(results) == 0):
            continue
        # Compare all results covariance between all parameters
        for fun, res in results.items():
            if (res[1][0:1].sum() < result[1][0:1].sum()):
                result = res
                func = fun

        # Check that width and peak are respected with a tolerance of 10%
        res_func = func(x, result[0][0], result[0][1], result[0][2])
        if(do_plot):
            plt.plot(x[peak], y[peak], "o", color="orange", label="Peak position @ {}".format(peak))
            plt.plot(x, res_func, linestyle="--")
            plt.title("Peak fitting on Raman spectra for sample {}".format(index))
            plt.xlabel("Raman shift ($cm^{-1}$")
            plt.ylabel("Intensity (Arbitrary units)")


@add_doc(
    '''

    Cuts the spectra/spectrum specified.

    
    Parameters
    ----------
    spectra: np.ndarray
    roi: (tuple, array, np.ndarray)
    index: (int, tuple, array, np.ndarray)
        Index/indices to preprocess.


    Returns
    -------
    The resulting copy of the cut spectra/spectrum.

    '''
)
def _cutSpectrum(spectra, roi, index=-1):
    spectra_c = spectra.__class__()

    index_strt = np.abs(spectra.wavenumbers-roi[0]).argmin()
    index_end = np.abs(spectra.wavenumbers-roi[1]).argmin()
    new_wvnmbr = spectra.wavenumbers[index_strt:index_end]

    if(index == -1):  # All signals
        for i in spectra.index:
            new_sig = spectra.intensity[i][index_strt:index_end]
            spectra_c.addSpectrum(new_wvnmbr, new_sig)
    else:
        if(isinstance(index, (tuple, list, np.ndarray))):  # Multiple signals
            for i in index:
                new_sig = spectra.intensity[i][index_strt:index_end]
                spectra_c.addSpectrum(new_wvnmbr, new_sig)
        elif(isinstance(index, int)):  # Only 1 signal
                new_sig = spectra.intensity[index][index_strt:index_end]
                spectra_c.addSpectrum(new_wvnmbr, new_sig)

    return spectra_c


@add_doc(
    '''
        Subtract the background having the background in the Spectra object.

        
        Parameters
        ----------
        spectra: np.ndarray
            Spectra object.
        index_baseline: int
            Index position of the background signal.
        index: (int, tuple, list, np.ndarray)
            Indices/indices of the signals that are going to be substracted the signal form the index_baseline param.
        inplace: bool
            Perform the change in the Spectra object (True) or create a new one and return (False)


        Returns
        -------
        if(inplace):
            None
        else:
            New Spectra object with preprocessed signal
    '''
)
def _removeBackground(spectra, index_baseline, index=-1, inplace=False):
    if inplace:
        spectra_c = spectra
    else:
        spectra_c = spectra.copy()

    if(index == -1):  # All signals
        for i in spectra_c.index:
            if i == index_baseline:
                pass
            new_sig = spectra_c.loc[i].intensity - spectra_c.loc[index_baseline]
            spectra_c.at[i, 'intensity'] = new_sig
    else:
        if(isinstance(index, tuple)):  # Multiple signals
            for i in index:
                if i == index_baseline:
                    pass
                new_sig = spectra_c.loc[i].intensity - spectra_c.loc[index_baseline]
                spectra_c.at[i, 'intensity'] = new_sig
        elif(isinstance(index, int)):  # Only 1 signal
            new_sig = spectra_c.loc[index].intensity - spectra_c.loc[index_baseline]
            spectra_c.at[index, 'intensity'] = new_sig

    if not inplace:
        return spectra_c


@add_doc(
    '''
    Removes the spikes from the spectrum/spectra specified using a median filter.


    Parameters
    ----------
    spectra: np.ndarray
    index: (int, tuple, list, np.ndarray)
        Index/indices to preprocess.
    inplace: bool
            Perform the change in the Spectra object (True) or create a new one and return (False)


    kwargs
    ------
    volume : array_like
        An N-dimensional input array.
    kernel_size : array_like, optional
        A scalar or an N-length list giving the size of the median filter window in each dimension. \
        Elements of kernel_size should be odd. If kernel_size is a scalar, then this scalar is used as \
        the size in each dimension. Default size is 3 for each dimension.


    Returns
    -------
        if(inplace):
            None
        else:
            New Spectra object with preprocessed signal

    '''
)
def _removeSpikes(spectra, index=-1, inplace=False, **kwargs):
    if inplace:
        spectra_c = spectra
    else:
        spectra_c = spectra.copy()

    if(index == -1):  # All signals
        for i in spectra_c.index:
            new_sig = medfilt(spectra_c.loc[i].intensity, **kwargs)
            spectra_c.at[i, 'intensity'] = new_sig
    else:
        if(isinstance(index, tuple)):  # Multiple signals
            for i in index:
                new_sig = medfilt(spectra_c.loc[i].intensity, **kwargs)
                spectra_c.at[i, 'intensity'] = new_sig
        elif(isinstance(index, int)):  # Only 1 signal
            new_sig = medfilt(spectra_c.loc[i].intensity, **kwargs)
            spectra_c.at[index, 'intensity'] = new_sig

    if not inplace:
        return spectra_c

@add_doc(
    '''

    Funtion to plot the different classes.

    '''
)
def _classDifferences(spectra, to_predict):
    plt.figure(figsize=(14,20))
    ax1 = plt.subplot(311)
    classes = np.unique(to_predict)
    for clss in classes:
        ax1.plot(spectra.wavenumbers, mode(spectra.intensity[(to_predict == clss).reshape(-1)]).mode[0], label=clss)
    ax1.set_title("Mode of the Raman spectra")
    ax1.set_xlabel("Raman shift ($cm^{-1}$)")
    ax1.set_ylabel("Intensity (Arbitrary units)")
    ax1.set_xlim([spectra.wavenumbers.min(), spectra.wavenumbers.max()])
    ax1.legend()
    ax2 = plt.subplot(312)
    for clss in classes:
        ax2.plot(spectra.wavenumbers, np.mean(spectra.intensity[(to_predict == clss).reshape(-1)], axis=0), label=clss)
    ax2.set_title("Mean of the Raman spectra")
    ax2.set_xlabel("Raman shift ($cm^{-1}$)")
    ax2.set_ylabel("Intensity (Arbitrary units)")
    ax2.set_xlim([spectra.wavenumbers.min(), spectra.wavenumbers.max()])
    ax2.legend()
    ax3 = plt.subplot(313)
    for clss in classes:
        ax3.plot(spectra.wavenumbers, spectra.intensity[(to_predict == clss).reshape(-1)].astype(np.float64).std(axis=0), label=clss)
    ax3.set_title("Std. Dev. of the Raman spectra")
    ax3.set_xlabel("Raman shift ($cm^{-1}$)")
    ax3.set_ylabel("Intensity (Arbitrary units)")
    ax3.set_xlim([spectra.wavenumbers.min(), spectra.wavenumbers.max()])
    ax3.legend()