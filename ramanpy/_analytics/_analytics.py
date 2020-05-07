 # -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:37:19 2020

@author: Francis Santos
"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm
from sklearn.cross_decomposition.pls_ import _PLS
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, \
                                    learning_curve, validation_curve, \
                                    GridSearchCV
from sklearn.decomposition import PCA
import warnings
import matplotlib.pyplot as plt
import math
from pandas import DataFrame


class PLSReg(_PLS):
    def __init__(self, n_components=2, *, scale=True,
                 max_iter=500, tol=1e-06, copy=True, algorithm="svd"):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="regression", mode="A",
            norm_y_weights=False, max_iter=max_iter, tol=tol,
            copy=copy, algorithm=algorithm)


def _testRegressors(spectra, to_predict, multithread, dim_red_only):
    '''

    Test multiple regressors (Support Vector Machines, Decision Tree,
    and K-Nearest Neighbor using the scikit-learn library. The data is divided
    into four parts:
        - Unaltered data
        - Normalized data
        - Standardized data
        - Data passed through a PowerTransformer

    The altered data is then passed through a PCA (Principal Component Analysis
    algorithm), which reduces the dimensions of the data. Both, dimension
    reduced and input data ar kept. Then they're fed to the different
    algorithms in all the forms.


    Parameters
    ----------
        spectra : Spectra
            Spectra object that contains all samples.
        to_predict : ndarray
            Array that contains the data of interest.
        multithread : bool
            If enabled, when performing GridSearchCV n_jobs = -1


    Returns
    -------
        chosen_regressor : ndarray
            A 3 columns array that contains information of the best regressor
            [
                    0 -> Score of the best regressor
                    1 -> Regressor object
                    2 -> Dictionary containing extra parameters {"name": Name \
                    of the algorithm, includes if PCA was used or not \
                    and the format of the data (i.e. normalized), "pca": PCA
                    used to reduced the dimensions.}
            ]


    '''
    # Check types are correct and arrays are the same size
    if("Spectra" not in spectra.__class__.__name__):
        raise AttributeError("The spectra attribute must be of Spectra type.")
    if(not isinstance(to_predict, np.ndarray)):
        raise AttributeError("The to_predict attribute must be a numpy array.")
    if(spectra.shape[0] != to_predict.shape[0]):
        raise ValueError(f"Shape mismatch: value array of size \
                         ({spectra.shape[0]}) and ({to_predict.shape[0]}) \
                         don't match.")

    # Results dictionary
    results = {}

    # Perform scaling
    print("Performing scaling...")
    spectra_minmax = spectra.copy()
    minmax = MinMaxScaler()
    minmax.fit(spectra_minmax.intensity.tolist())
    spectra_minmax.intensity[:] = minmax.fit_transform(spectra_minmax.intensity)
    # for i in spectra.index:
    #     new_sig = minmax.transform(spectra_minmax.intensity[i].reshape(1,-1))
    #     spectra_minmax.intensity[i] = new_sig[0]

    # Stack data in order to comply with scikit inputs
    X = np.stack(spectra_minmax.intensity)

    # Perform dimensionality reduction
    print("Performing dimensionality reduction...")
    pca_minmax = PCA()
    pca_minmax.fit(X)  # Norm
    n_components_minmax = pca_minmax.explained_variance_ratio_.size
    for i in range(1, n_components_minmax):
        expl_var = pca_minmax.explained_variance_ratio_[0:i+1].sum()
        if(expl_var > 0.95):
            n_components_minmax = i
            pca_minmax = PCA(n_components=n_components_minmax)
            pca_minmax.fit(X)
            break
    spectra_minmax_pca = pca_minmax.transform(X)

    # Create regressors
    print("Creating regressors...")
    rdg = Ridge()
    pls = PLSReg()
    svr = svm.LinearSVR()
    print("Testing regressors...")
    warnings.filterwarnings("ignore")

    results["RDG_minmax_PCA"] = _doModelSelection(rdg, spectra_minmax_pca, to_predict, multithread)
    results["SVR_minmax_PCA"] = _doModelSelection(svr, spectra_minmax_pca, to_predict, multithread)
    results["PLS_minmax_PCA"] = _doModelSelection(pls, spectra_minmax_pca, to_predict, multithread)
    if(not dim_red_only):
        results["RDG_minmax"] = _doModelSelection(rdg, X, to_predict, multithread)
        results["PLS_minmax"] = _doModelSelection(pls, X, to_predict, multithread)
        results["SVR_minmax"] = _doModelSelection(svr, X, to_predict, multithread)

    # Plot resulting scores
    print("Plotting resuling scores of the RMSECV")
    plt.figure(figsize=(18, 12))
    for name, result in results.items():
        score_index = 0
        plt.bar(name, math.sqrt(abs(result[score_index])))
    plt.xlabel("Model")
    plt.ylabel("Score (RMSECV)")
    plt.show()
    # Evaluate all regresors
    typ = "regressor"
    first = True
    for name, result in results.items():
        # Check PCA
        if("minmax_PCA" in name):
            pca = pca_minmax
        else:
            pca = None
        # Fill variable first if not set
        if(first):
            chosen_reg = (result[0], result[1], {"name": name, "pca": pca, "type": typ})
            first = False
            continue
        # Compare
        if(abs(result[0]) < abs(chosen_reg[0])):
            chosen_reg = (result[0], result[1], {"name": name, "pca": pca, "type": typ})

    # Return model of best regressor with the results
    return chosen_reg


def _testClassifiers(spectra, to_predict, multithread, dim_red_only):
    pass


def _doModelSelection(estimator, X, Y, multithread):
    n_jobs = -1 if(multithread) else None
    print("Evaluating...")
    if(isinstance(estimator, Ridge)):
        estimator = Ridge()
        score_metrics = ["r2", "neg_mean_squared_error"]
        refit = "neg_mean_squared_error"
        params = {
                'alpha': np.logspace(-1, 4, 50)
                }
    elif(isinstance(estimator, PLSReg)):
        n_components = X.shape[1] if X.shape[1] < 50 else 50
        n_components_to_test = X.shape[1] if X.shape[1] < 10 else 10
        estimator = PLSReg()
        score_metrics = ["r2", "neg_mean_squared_error"]
        refit = "neg_mean_squared_error"
        params = {
                'n_components': np.linspace(1, n_components, n_components_to_test, endpoint=True, dtype=np.int8),
                'scale': [False, True],
                'algorithm': ["svd", "nipals"]
                }
    elif(isinstance(estimator, svm.LinearSVR)):
        estimator = svm.LinearSVR(max_iter=750)
        score_metrics = ["r2", "neg_mean_squared_error"]
        refit = "neg_mean_squared_error"
        params = {
                'epsilon': np.linspace(0.0, 1.0, 11, endpoint=True),
                'C': np.linspace(0.01, 1.0, 11, endpoint=True)
                }
    elif(isinstance(estimator,KNeighborsClassifier)):
        estimator = KNeighborsClassifier(n_jobs=-1)
        score_metrics = "f1_weighted"
        refit = "f1_weighted"
        params = {
                'n_neighbors': np.linspace(1, 3, 3, endpoint=True,
                                        dtype=np.int8),
                'weights': ("uniform", "distance"),
                'leaf_size': np.linspace(1, 50, 3, endpoint=True,
                                        dtype=np.int8),
                'metric': ["euclidean", "manhattan", "chebyshev"]
                }
    elif(isinstance(estimator, svm.SVC)):
        estimator = svm.SVC(gamma='scale')
        score_metrics = "f1_weighted"
        refit = "f1_weighted"
        params = {
                'C': np.linspace(1, 1e8, 10, endpoint=True, dtype=np.int8),
                'kernel': ['linear']
                }
    elif(isinstance(estimator, DecisionTreeClassifier)):
        estimator = DecisionTreeClassifier()
        score_metrics = "f1_weighted"
        refit = "f1_weighted"
        params = {
                'criterion': ["gini", "entropy"],
                'max_depth': np.linspace(1, 33, 3, endpoint=True),
                'min_samples_split': np.linspace(0.1, 1.0, 3, endpoint=True),
                'min_samples_leaf': np.linspace(0.1, 0.5, 3, endpoint=True)
                }
    else:
        raise AttributeError("The type of classifier/regressor is \
                            not supported.")
    clf = GridSearchCV(estimator, params, cv=10, error_score=np.nan,
                    scoring=score_metrics, refit=refit, n_jobs=n_jobs)
    clf.fit(X, Y)
    print("Evaluating done.")

    # Keep that with lower mean AND std. dev. error
    results_cv = DataFrame(clf.cv_results_)
    scores_mean_std = abs(results_cv["mean_test_neg_mean_squared_error"]) + abs(results_cv["std_test_neg_mean_squared_error"])
    index_best_score = scores_mean_std.argmin()
    estimator.set_params(**results_cv.at[index_best_score, "params"])
    # return clf.best_score_, clf.best_estimator_
    return scores_mean_std[index_best_score], estimator


def _trainModel(spectra, to_predict):
    # Obtain information from model structure
    model = spectra._model[1]
    model_info = spectra._model[2]
    data = np.stack(spectra.intensity)
    data_aux = data.copy()

    # Scale if in model (for learning_curve)
    if("minmax" in model_info["name"]):
        scaler = MinMaxScaler()
        data_aux = scaler.fit_transform(data_aux)
    if("PCA" in model_info["name"]):        
        pca = model_info["pca"]
        data_aux = pca.fit_transform(data_aux)
        
    scoring = "neg_mean_squared_error"

    # Scoring metric
    if(model_info["type"] == "regression"):
        scoring = "neg_mean_squared_error"
    elif(model_info["type"] == "classification"):
        scoring = "f1_weighted"

    # Get optimum test size based on the learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, data_aux, to_predict, cv=10, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    diff_scores = (abs(test_scores_mean) + abs(test_scores_std) - abs(train_scores_mean) - abs(train_scores_std))
    
    min_diff_score = train_sizes[diff_scores.argmin()]

    plt.figure(figsize=(15,12))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="b")
    plt.plot(train_sizes, train_scores_mean, "o-", color="red", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="blue", label="Cross-validation score")

    plt.title("Learning Curve")
    plt.xlabel("Number of training samples")
    plt.ylabel("Negative Mean Squared Error")
    plt.legend()
    plt.show()
    
    # Process data accordingly, first split
    X, X_test, Y, Y_test = train_test_split(data, to_predict, test_size=1 - min_diff_score/data.shape[0],
                                            shuffle=True, random_state=0)
    
    # Scale if in model
    if("minmax" in model_info["name"]):
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)
        spectra._model[2]["scaler"] = scaler
    if("PCA" in model_info["name"]):        
        pca = model_info["pca"]
        X = pca.fit_transform(X)
        X_test = pca.transform(X_test)
    
    # Fit to model    
    model.fit(X, Y)

    # Predict train and test sets
    Y_train_pred = model.predict(X)
    Y_test_pred = model.predict(X_test)
    return {"Y_train": Y,
            "Y_test": Y_test,
            "Y_test_pred": Y_test_pred,
            "Y_train_pred": Y_train_pred}


def _predict(spectra, index):
    model = spectra._model[1]
    model_info = spectra._model[2]

    if(index != -1 and not isinstance(index, int)):
        data = np.stack(spectra.intensity[index].copy())
    elif(index == -1):
        data = np.stack(spectra.intensity.copy())
    elif(isinstance(index, int)):
        data = np.stack(spectra.intensity[index].copy()).reshape(1, -1)

    # Scale if in model
    if("std" in model_info["name"]):
        scaler = model_info["scaler"]
        data = scaler.transform(data)
    elif("minmax" in model_info["name"]):
        scaler = model_info["scaler"]
        data = scaler.transform(data)
    if("PCA" in model_info["name"]):        
        pca = model_info["pca"]
        data = pca.transform(data)

    return model.predict(data)
