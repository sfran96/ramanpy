# -*- coding: utf-8 -*-
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


# 1 Gaussian form
def _1gaussian(x, intensity, center, width):
    return intensity * np.exp(-(x-center)**2 / width)


# 2 Gaussian form
def _2gaussian(x, intensity1, center1, width1, intensity2, center2, width2):
    return intensity1 * np.exp(-(x-center1)**2 / width1) + \
            intensity2 * np.exp(-(x-center2)**2 / width2)


# 1 Lorentzian form
def _1lorentzian(x, intensity, center, width):
    return intensity * 1/(1 + ((x-center) / width)**2)


# 2 Lorentzian form
def _2lorentzian(x, intensity1, center1, width1, intensity2, center2, width2):
    return intensity1 * 1/(1 + ((x-center1) / width1)**2) + \
            intensity2 * 1/(1 + ((x-center2) / width2)**2)


# Voigtian form
def _1voigtian(x, intensity, center, width, lor_gaus):
    return intensity * (lor_gaus * (_1gaussian(x, intensity, center, width)/intensity) + (1-lor_gaus) * (_1lorentzian(x, intensity, center, width)/intensity))


# PearsonIV
def _1pearsonIV(x, intensity, position, width, shape_param1, shape_param2):
    return intensity * (1 + ((x - position) / width)**2)**(-shape_param1) * (np.exp(-shape_param2 * np.arctan((x - position) / width)))


def _removeBaseline(spectra, roi, method, index=-1, inPlace=False, **kwargs):
    if inPlace:
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
        if(isinstance(index, tuple)):  # Multiple signals
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

    if not inPlace:
        return spectra_c


def _smoothSignal(spectra, index=-1, method="flat", inPlace=False,
                  **kwargs):
    if inPlace:
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

    if not inPlace:
        return spectra_c


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
            plt.xlabel("Raman shift")
            plt.ylabel("Intensity")


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
        if(isinstance(index, tuple)):  # Multiple signals
            for i in index:
                new_sig = spectra.intensity[i][index_strt:index_end]
                spectra_c.addSpectrum(new_wvnmbr, new_sig)
        elif(isinstance(index, int)):  # Only 1 signal
                new_sig = spectra.intensity[index][index_strt:index_end]
                spectra_c.addSpectrum(new_wvnmbr, new_sig)

    return spectra_c


def _removeBackground(spectra, index_baseline, index=-1, inPlace=False):
    if inPlace:
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

    if not inPlace:
        return spectra_c


def _removeSpikes(spectra, index=-1, inPlace=False, **kwargs):
    if inPlace:
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

    if not inPlace:
        return spectra_c

def _classDifferences(spectra, to_predict):
    plt.figure(figsize=(14,20))
    ax1 = plt.subplot(311)
    classes = np.unique(to_predict)
    for clss in classes:
        ax1.plot(spectra.wavenumbers, mode(spectra.intensity[(to_predict == clss).reshape(-1)]).mode[0], label=clss)
    ax1.set_title("Mode of the Raman spectra")
    ax1.set_xlabel("Raman shift")
    ax1.set_ylabel("Intensity")
    ax1.set_xlim([spectra.wavenumbers.min(), spectra.wavenumbers.max()])
    ax1.legend()
    ax2 = plt.subplot(312)
    for clss in classes:
        ax2.plot(spectra.wavenumbers, np.mean(spectra.intensity[(to_predict == clss).reshape(-1)], axis=0), label=clss)
    ax2.set_title("Mean of the Raman spectra")
    ax2.set_xlabel("Raman shift")
    ax2.set_ylabel("Intensity")
    ax2.set_xlim([spectra.wavenumbers.min(), spectra.wavenumbers.max()])
    ax2.legend()
    ax3 = plt.subplot(313)
    for clss in classes:
        ax3.plot(spectra.wavenumbers, spectra.intensity[(to_predict == clss).reshape(-1)].astype(np.float64).std(axis=0), label=clss)
    ax3.set_title("Std. Dev. of the Raman spectra")
    ax3.set_xlabel("Raman shift")
    ax3.set_ylabel("Intensity")
    ax3.set_xlim([spectra.wavenumbers.min(), spectra.wavenumbers.max()])
    ax3.legend()