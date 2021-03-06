﻿# Ramanpy

## Contained modules
[![Python version](https://img.shields.io/badge/python-3.7.4-blue)]() [![Numpy Requirement](https://img.shields.io/badge/numpy-1.16.5-green)](https://github.com/numpy/numpy) [![Pandas Requirement](https://img.shields.io/badge/pandas-1.0.3-green)](https://github.com/pandas-dev/pandas) [![Matplotlib Requirement](https://img.shields.io/badge/matplotlib-3.2.1-green)](https://github.com/matplotlib/matplotlib) [![Scipy Requirement](https://img.shields.io/badge/scipy-1.4.1-green)](https://github.com/scipy/scipy) [![Rampy Requirement](https://img.shields.io/badge/rampy-0.4.5-green)](https://github.com/charlesll/rampy) [![Scikit-Learn Requirement](https://img.shields.io/badge/sklearn-0.21.3-green)](https://github.com/scikit-learn/scikit-learn) [![SPC Requirement](https://img.shields.io/badge/spc-0.4-green)](https://github.com/rohanisaac/spc)

## Licensing and last status
[![License](https://img.shields.io/github/license/sfran96/ramanpy)](https://github.com/sfran96/ramanpy/blob/master/LICENSE) [![Last commit](https://img.shields.io/github/last-commit/sfran96/ramanpy)]() 

Ramanpy is a module that contains and used a collection of libraries to simplify the importing, pre-processing and analysis of Raman spectroscopy signals. It is distributed under the GNU GPL 2.0 license.

The software was developed in 2020 by Francis Santos as part of a master dissertation for the Master of Science in Biomedical Engineering at VUB/UGent.

## Installation
### Dependencies

Ramanpy requires:

- Python (>= 3.7)
- NumPy (>= 1.16.5)
- Scipy (>= 1.4.1)
- Rampy (>= 0.4.5)
- Pandas (>= 1.0.3)
- Scikit-Learn (>= 0.21.3)
- spc (>= 0.4)

### User installation
- Download the repository from GitHub
  - If using Git
    ```
    $ git clone https://github.com/sfran96/ramanpy
    ```
  - If using the GitHub interface, click "Code" followed by "Download ZIP"
- (If ZIP downloaded) Un-ZIP the files
- Navigate to the `ramanpy` folder and execute the following on a console:

    ```
    $ python setup.py build
    $ python setup.py install
    ```
    
    If using Anaconda Environment:
    
    ```
    $ conda install conda-build
    $ conda develop .
    ```

### Development
A simple Python Notebook using this module to read an SPC file would look like:
```python
import ramanpy as rp

# Create containing object
spectra = rp.Spectra()
# Read from file and place it in previously created object
rp.readFile("folder/file.spc", spectra)
```
