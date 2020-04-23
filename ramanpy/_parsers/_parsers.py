# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:20:10 2020

@author: frabb
"""
__all__ = ['_readCSV', '_readSPC', '_readMATLAB', '_readJCAMP', '_setPreValues', '_removePreValues']
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from scipy.io import loadmat
import sys as sys
import spc as spc
import copy as _copy
from pyjdxx import jdx
from jcamp import JCAMP_reader


class config:
    _finished = False
    _index1 = None
    _index2 = None
    _index3 = None
    _multicolumn = False
    _set_values = False


def _setPreValues():
    config._set_values = True


def _removePreValues():
    config._finished = False
    config._index1 = None
    config._index2 = None
    config._index3 = None
    config._multicolumn = False
    config._set_values = False


def _readCSV(path, spectra, with_to_predict=False, **kwargs):
    # Control variables
    finished = False or config._finished
    index1 = None or config._index1
    index2 = None or config._index2
    index3 = None or config._index3
    multicolumn = False or config._multicolumn

    samples = pd.read_csv(path, **kwargs)
    i = 5 if len(samples.columns) > 10 else False  # Checking if there are more
    # than 10 columns to show, if so show them all without "..."


    if(not config._finished):
        # Print overhead
        table_columns = PrettyTable()
        if(i):
            table_columns.field_names = np.append(samples.columns[0:i],
                                                  np.append(["..."],
                                                            samples.columns[-i:]))
        else:
            table_columns.field_names = samples.columns
        print(table_columns)
        # Do-while the answer is correct
        inp = input("Are the X values distributed in multiple columns? Yes/No [Yes]: ") or "Yes"  
        while not finished:
            if(inp.lower() == "yes"):
                multicolumn = True
                inp = input("Start column of X values? ")
                while not finished:
                    if(inp in samples.columns):
                        index1 = inp
                        break
                    inp = input("Start column of X values? ")
                inp = input("End column of X values? ")
                while not finished:
                    if(inp in samples.columns):
                        index2 = inp
                        finished = True
                        break
                    inp = input("End column of X values? ")
                pass
            elif(inp.lower() == "no"):
                inp = input("Column that contains the wavenumbers? ")
                while not finished:
                    if(inp in samples.columns):
                        index1 = inp
                        break
                    inp = input("Column that contains the wavenumbers? ")
                inp = input("Column that contains the intensities? ")
                while not finished:
                    if(inp in samples.columns):
                        index2 = inp
                        break
                    inp = input("Column that contains the intensities? ")
            else:
                inp = input("Are the X values distributed in multiple columns? Yes/No [No]: ")
                
        if(with_to_predict):
            inp = input("Name of column containing the to-predict values: ")
            while(not finished):  # If there's more than one value ask for a specific one:
                if(inp in samples.columns):
                    index3 = np.where(samples.columns == inp)
                    to_predict = samples.loc[:, index3].to_numpy()
                    finished = True
                    break
                inp = input("Try again: ")
        else:
            finished = True

    # If the wavenumbers are distributed in columns
    if multicolumn:
        wvnmbrs = samples.loc[:, index1:index2].columns.to_numpy() if not isinstance(samples.loc[:, index1:index2].columns.to_numpy()[0], str) else np.array(range(0, len(samples.loc[:, index1:index2].columns.to_numpy())))
        intensities = samples.loc[:, index1:index2].to_numpy()
        for intensity in intensities:
            spectra.addSpectrum(wvnmbrs, intensity, path)

    else:
        wvnmbrs = samples.loc[:, index1].to_numpy()
        intensities = samples.loc[:, index2].to_numpy()
        spectra.addSpectrum(wvnmbrs, intensities, path)

    if(config._set_values):
        config._index1 = index1
        config._index2 = index2
        config._index3 = index3
        config._finished = finished
        config._multicolumn = multicolumn

    if(with_to_predict):
        return to_predict


def _readMATLAB(path, spectra, with_to_predict=False, **kwargs):
    file = loadmat(path, **kwargs)
    finished = False or config._finished
    index1 = None or config._index1
    index2 = None or config._index2
    index3 = None or config._index3

    # Ask for beginning of importing frequency
    if(not finished):
        table_columns = PrettyTable()
        table_columns.field_names = np.append(file['VarLabels'][0:5],
                                              np.append(["..."],
                                                        file['VarLabels'][-5:]))
        print(table_columns)
        inp = input("Beginning of importing frequency: ")
        while(not finished):  # If there's more than one value ask for a specific one:
            if(inp in file['VarLabels']):
                index1 = np.where(file['VarLabels'] == inp)
                break
            inp = input("Try again: ")

        # Ask for end of importing frequency
        inp = input("End of importing frequency: ")
        while(not finished):  # If there's more than one value ask for a specific one:
            if(inp in file['VarLabels']):
                index2 = np.where(file['VarLabels'] == inp)
                break
            inp = input("Try again: ")

        if(with_to_predict):
            inp = input("Name of column containing the to-predict values: ")
            while(not finished):  # If there's more than one value ask for a specific one:
                if(inp in file['VarLabels']):
                    index3 = np.where(file['VarLabels'] == inp)
                    to_predict = []
                    for row in file['Matrix']:
                        to_predict.append(row[index3[0][0]])
                    to_predict = np.array(to_predict)
                    finished = True
                    break
                inp = input("Try again: ")
        else:
            finished = True

    wvnmbrs = file['VarLabels'][index1[0][0]:index2[0][0]].astype(np.float)        

    flipped = False
    # Reverse wavenumbers if they're countdown
    if(wvnmbrs[0] > wvnmbrs[-1]):
        wvnmbrs = np.flip(wvnmbrs)
        flipped = True

    # Introduce all inputs in the spectra
    for row in file['Matrix']:
        intensity = row[index1[0][0]:index2[0][0]]
        if(flipped):
            intensity = np.flip(intensity)
        spectra.addSpectrum(wvnmbrs, intensity, path)

    if(config._set_values):
        config._index1 = index1
        config._index2 = index2
        config._index2 = index3
        config._finished = finished

    if(with_to_predict):
        return to_predict


def _readSPC(path, spectra, **kwargs):
    # Cancelout output of this module
    _stdout = _copy.copy(sys.stdout)
    sys.stdout = None
    spc_file = spc.File(path, **kwargs)
    # Restart standard output
    sys.stdout = _stdout

    wvnmbrs = spc_file.x
    # Append the new spectrum/spectra
    for signal in spc_file.sub:
        spectra.addSpectrum(wvnmbrs, signal.y, path)


def _readJCAMP(path, spectra, **kwargs):
    try:
        first_reader = True
        file = jdx.jdx_file_reader(path, **kwargs)
    except Exception:
        file = JCAMP_reader(path, **kwargs)
        first_reader = False

    if(first_reader):
        if file.size == 1:
            spectra.addSpectrum(file[0]['x'], file[0]['y'], path)
        elif file.size > 1:
            for spectrum in file:
                spectra.addSpectrum(spectrum['x'], spectrum['y'], path)
    else:
        spectra.addSpectrum(file['x'], file['y'], path)
