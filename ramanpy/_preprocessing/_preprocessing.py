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
            new_sig, __ = rp.baseline(spectra_c.loc[i, "wavenumbers"],
                                      spectra_c.loc[i, "intensity"], roi,
                                      method, **kwargs)
#            if(new_sig.min() < 0):
#                new_sig -= new_sig.min()
            spectra_c.at[i, 'intensity'] = new_sig.reshape(-1,)
    else:
        if(isinstance(index, tuple)):  # Multiple signals
            for i in index:
                new_sig, __ = rp.baseline(spectra_c.loc[i, "wavenumbers"],
                                          spectra_c.loc[i, "intensity"], roi,
                                          method, **kwargs)
#                if(new_sig.min() < 0):
#                    new_sig -= new_sig.min()
                spectra_c.at[i, 'intensity'] = new_sig.reshape(-1,)
        elif(isinstance(index, int)):  # Only 1 signal
            new_sig, __ = rp.baseline(spectra_c.loc[index, "wavenumbers"],
                                      spectra_c.loc[index, "intensity"], roi,
                                      method, **kwargs)
#            if(new_sig.min() < 0):
#                new_sig -= new_sig.min()
            spectra_c.at[index, 'intensity'] = new_sig.reshape(-1,)

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
            new_sig = rp.smooth(spectra_c.loc[i, "wavenumbers"],
                                spectra_c.loc[i, "intensity"],
                                method, **kwargs)
            spectra_c[i, 'intensity'] = new_sig[0]
    else:
        if(isinstance(index, tuple)):  # Multiple signals
            for i in index:
                new_sig = rp.smooth(spectra_c.loc[i, "wavenumbers"],
                                    spectra_c.loc[i, "intensity"],
                                    method, **kwargs)
                spectra_c[i, 'intensity'] = new_sig[0]
        elif(isinstance(index, int)):  # Only 1 signal
            new_sig = rp.smooth(spectra_c.loc[index, "wavenumbers"],
                                spectra_c.loc[index, "intensity"],
                                method, **kwargs)
            spectra_c[index, 'intensity'] = new_sig[0]

    if not inPlace:
        return spectra_c


def _detectPeaks(spectra, index=0, do_plot=True):
    # Obtain the full signals in a way they can be accessed more easily later
    x = spectra.loc[index].wavenumbers
    y = spectra.loc[index].intensity

    # Find peaks and information related to it (width, floor, start and end)
    peaks, __ = find_peaks(y, prominence=y.mean()/(x.size/100))
    peak_widths_info = peak_widths(y, peaks, rel_height=1)
    widths = peak_widths_info[0]
    widths_pos_init = peak_widths_info[2]
    widths_pos_end = peak_widths_info[3]

    if(do_plot):
        plt.figure(figsize=(16, 16))
        plt.plot(x, y, linewidth=2)

    # For every peak with these functions
    funcs = (_1gaussian, _1lorentzian)
    for i in range(0, len(peaks)):
        results = {}
        peak = peaks[i]
        initial_fitting = np.array([y[peak], x[peak], widths[i]])

        if x[int(widths_pos_init[i]):int(widths_pos_end[i])].size < 3:
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
            plt.plot(x[peak], y[peak], "*", color="orange")
#            plt.plot(x, res_func, linestyle=":")


def _cutSpectrum(spectra, roi, index=-1, inPlace=False):
    if inPlace:
        spectra_c = spectra
    else:
        spectra_c = spectra.copy()

    if(index == -1):  # All signals
        for i in spectra_c.index:
            index_strt = np.abs(spectra.loc[i].wavenumbers-roi[0]).argmin()
            index_end = np.abs(spectra.loc[i].wavenumbers-roi[1]).argmin()
            new_sig = spectra_c.loc[i].intensity[index_strt:index_end]
            new_wvnmbr = spectra_c.loc[i].wavenumbers[index_strt:index_end]
            spectra_c.at[i, 'intensity'] = new_sig
            spectra_c.at[i, 'wavenumbers'] = new_wvnmbr
    else:
        if(isinstance(index, tuple)):  # Multiple signals
            for i in index:
                index_strt = np.abs(spectra.loc[i].wavenumbers-roi[0]).argmin()
                index_end = np.abs(spectra.loc[i].wavenumbers-roi[1]).argmin()
                new_sig = spectra_c.loc[i].intensity[index_strt:index_end]
                new_wvnmbr = spectra_c.loc[i].wavenumbers[index_strt:index_end]
                spectra_c.at[i, 'intensity'] = new_sig
                spectra_c.at[i, 'wavenumbers'] = new_wvnmbr
        elif(isinstance(index, int)):  # Only 1 signal
            index_strt = np.abs(spectra.loc[index].wavenumbers-roi[0]).argmin()
            index_end = np.abs(spectra.loc[index].wavenumbers-roi[1]).argmin()
            new_sig = spectra_c.loc[index].intensity[index_strt:index_end]
            new_wvnmbr = spectra_c.loc[index].wavenumbers[index_strt:index_end]
            spectra_c.at[index, 'intensity'] = new_sig
            spectra_c.at[index, 'wavenumbers'] = new_wvnmbr

    if not inPlace:
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
