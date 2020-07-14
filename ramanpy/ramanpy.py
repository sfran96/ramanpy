"""

Created on April 2020

@author: Francis Santos

"""
from __future__ import division, print_function
__all__ = ['Spectra', 'readFile', 'readFiles']
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from glob import glob as _glob
import numpy as np
from _parsers import _readCSV, _readJCAMP, _readMATLAB, _readSPC, _setPreValues, _removePreValues
from _preprocessing import _removeBaseline, _smoothSignal, _removeBackground, _detectPeaks, _cutSpectrum, _removeSpikes, _classDifferences
from _analytics import _testClassifiers, _testRegressors, _trainModel, _predict, _resultsIsolatedFrequencies
import pickle
from datetime import datetime
from pathlib import Path
from utils import add_doc, copy_doc_string


class Spectra(DataFrame):

    @property
    def _constructor(self):
        return Spectra

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self._model = None
        if("max_decimals" in kwargs):
            self._max_decimals = kwargs["max_decimals"]
            del(kwargs["max_decimals"])
        else:
            self._max_decimals = 1

    @property
    def wavenumbers(self):
        return self.columns.values.astype(np.float)

    @property
    def intensity(self):
        return self.values

    @add_doc(
        '''
        
        Function used to add a spectrum to the Spectra DataFrame.


        Parameters
        ----------
            wavenumbers : ndarray
                Array that contains the raman shifts.
            intensity : ndarray
                Array that contains the intensities for each raman shift.


        Returns
        -------
            None


        '''
    )
    def addSpectrum(self, wavenumbers, intensity):
        if(len(self.index) == 0):
            self.__dict__.update(pd.DataFrame(columns=np.round(wavenumbers, self._max_decimals).astype("str")).__dict__)
            self.at[0] = intensity  # adding a row
        else:
            self.at[self.index.max() + 1] = intensity  # adding a row
        self.sort_index(inplace=True)

    @copy_doc_string(_removeBaseline)
    def removeBaseline(self, roi, method, index=-1, inplace=False, **kwargs):
        result = _removeBaseline(self, roi, method, index, inplace, **kwargs)
        if(not inplace):
            return result

    @copy_doc_string(_smoothSignal)
    def smoothSignal(self, index=-1, method="flat", inplace=False,
                     **kwargs):
        result = _smoothSignal(self, index, method, inplace, **kwargs)
        if(not inplace):
            return result

    @copy_doc_string(_detectPeaks)
    def detectPeaks(self, index=0, do_plot=True):
        _detectPeaks(self, index, do_plot)

    @copy_doc_string(_cutSpectrum)
    def cutSpectrum(self, roi, index=-1):
        result = _cutSpectrum(self, roi, index)
        return result

    @add_doc(
        '''
        '''
    )
    def plotSignal(self, index=0, figsize=(15,10)):
        plt.figure(figsize=figsize)
        ax = plt.gca()
        plt.plot(self.wavenumbers, self.intensity[index])
        plt.title(f"Raman signal for sample {index}")
        plt.xlabel("Raman shift")
        plt.ylabel("Intensity")
        plt.xlim(self.wavenumbers.min(), self.wavenumbers.max())
        plt.ylim(self.intensity[index].min(), self.intensity[index].max())
        return ax

    @copy_doc_string(_removeBackground)
    def removeBackground(self, index_baseline, index=-1, inplace=False):
        result = _removeBackground(self, index_baseline, index, inplace)
        if(not inplace):
            return result

    @copy_doc_string(_removeSpikes)
    def removeSpikes(self, index=-1, inplace=False, **kwargs):
        result = _removeSpikes(self, index, inplace)
        if(not inplace):
            return result

    @copy_doc_string(_testRegressors)
    def testRegressors(self, to_predict, multithread=False, **kwargs):
        self._model = _testRegressors(self, to_predict, multithread, **kwargs)

    @copy_doc_string(_testClassifiers)
    def testClassifiers(self, to_predict, multithread=False, **kwargs):
        self._model = _testClassifiers(self, to_predict, multithread, **kwargs)

    @add_doc(
        '''

        Function to save a trained model into the defined path in a Pickles wrapper.

        
        Parameters
        ----------
        path: str
            Path to save the file to. Default is "models".

        
        Returns
        -------
        None

        '''
    )
    def saveModel(self, path="models"):
        if(not isinstance(self._model, type(None))):
            Path(path).mkdir(parents=True, exist_ok=True)
            filename = datetime.today().strftime('%d-%m-%Y-%H%M')
            with open(f"{path}/{filename}.pkl", 'wb') as output:
                pickle.dump(self._model, output, pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError("There's no model to be saved at this moment.")

    @add_doc(
        '''

        Function to load a previously trained model into the defined path in a Pickles wrapper.

        
        Parameters
        ----------
        path: str
            Path to load the file from.

        
        Returns
        -------
        None

        '''
    )
    def loadModel(self, path_to_file):
        if(isinstance(path_to_file, str) and ".pkl" in path_to_file):
            with open(path_to_file, 'rb') as input:
                model = pickle.load(input)
                if(isinstance(model, tuple) and isinstance(model[2], dict)):
                    self._model = model
        else:
            raise AttributeError("The 'path_to_file' attribute must be of format *.pkl")

    @copy_doc_string(_trainModel)
    def trainModel(self, to_predict, show_graph=False, cv = 10):
        return _trainModel(self, to_predict, show_graph, cv)

    @copy_doc_string(_predict)
    def predictFromModel(self, index=0):
        return _predict(self, index)

    @copy_doc_string(_classDifferences)
    def classDifferences(self, to_predict):
        _classDifferences(self, to_predict)

    @copy_doc_string(_resultsIsolatedFrequencies)
    def resultsIsolatedFrequencies(self, to_predict, slots=10, cv=10):
        _resultsIsolatedFrequencies(self, to_predict, slots=slots, cv=cv)


@add_doc(
    '''

    Reads one file. Can read CSV, SPC and JCAMP-DX files.

    
    Parameters
    ----------
    path: str
        Path to the file of interest.
    spectra: Spectra
        Spectra object that will contain the newly read data.
    with_to_predict: bool
        If `True` the read files contain the reference value for training. Only read if type is CSV.
    

    kwargs
    ------
    Parameters for the different parsers.


    Returns
    -------
    The reference values if `with_to_predict = True`.
    '''
)
def readFile(path, spectra, with_to_predict=False, **kwargs):
    path = path.lower()

    # Comma-separeted-values file format
    if path.endswith(".csv") or path.endswith(".txt"):
        return _readCSV(path, spectra, with_to_predict, **kwargs)
    # SPC file format
    elif path.endswith(".spc"):
        _readSPC(path, spectra, **kwargs)
    # JCAMP JDX file format
    elif path.endswith(".jdx") or path.endswith(".jd"):
        _readJCAMP(path, spectra, **kwargs)
    # MATLAB file format
    elif path.endswith(".mat"):
        return _readMATLAB(path, spectra, with_to_predict, **kwargs)
    # REST of file formats
    else:
        raise AttributeError("""The format of the input file is not supported
                             by this software.""")


@add_doc(    
    '''

    Read multiple files with the same format from one specific folder. Can read CSV, SPC and JCAMP-DX files.

    
    Parameters
    ----------
    path: str
        Path of interest.
    spectra: Spectra
        Spectra object that will contain the newly read data.
    frmt: str
        File extension format.
            "csv" / "txt" for CSV files
            "spc" for SPC files
            "jdx" / "jd" for JDX files
    

    kwargs
    ------
    Parameters for the different parsers.


    Returns
    -------
    None
    '''
)
def readFiles(path, spectra, frmt="spc", **kwargs):
    files = _glob(f"{path}/*.{frmt}")
    for file_pathname in files:
        # If it's the same file change the configuration
        if(file_pathname == files[0]):
            _setPreValues()

        readFile(file_pathname, spectra, **kwargs)
        
        # Otherwise
        if(file_pathname == files[-1]):
            _removePreValues()
