 # -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:37:19 2020

@author: Francis Santos
"""
from sklearn.preprocessing import StandardScaler, PowerTransformer, Normalizer
from sklearn import svm
from sklearn.cross_decomposition.pls_ import _PLS
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, \
                                    learning_curve, validation_curve
from sklearn.decomposition import PCA
import warnings
import matplotlib.pyplot as plt


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

    # Perform normalization
    print("Performing normalization...")
    spectra_std = spectra.copy()
    spectra_pwr = spectra.copy()
    spectra_norm = spectra.copy()
    std_scaler = StandardScaler()
    pwr_trans = PowerTransformer()
    normalizer = Normalizer()
    std_scaler.fit(spectra_std.loc[:, 'intensity'].tolist())
    pwr_trans.fit(spectra_pwr.loc[:, 'intensity'].tolist())
    normalizer.fit(spectra_norm.loc[:, 'intensity'].tolist())

    for i in spectra.index:
        new_sig = std_scaler.transform(spectra_std.at[i, 'intensity'].reshape(1, -1))
        spectra_std.at[i, 'intensity'] = new_sig[0]
        new_sig = pwr_trans.transform(spectra_pwr.at[i, 'intensity'].reshape(1,-1))
        spectra_pwr.at[i, 'intensity'] = new_sig[0]
        new_sig = normalizer.transform(spectra_norm.at[i, 'intensity'].reshape(1,-1))
        spectra_norm.at[i, 'intensity'] = new_sig[0]

    # Stack data in order to comply with scikit inputs
    X_orig = np.stack(spectra.loc[:, 'intensity'])
    X_std = np.stack(spectra_std.loc[:, 'intensity'])
    X_pwr = np.stack(spectra_pwr.loc[:, 'intensity'])
    X_norm = np.stack(spectra_norm.loc[:, 'intensity'])

    # Perform dimensionality reduction
    print("Performing dimensionality reduction...")
    pca_std = PCA()
    pca_std.fit(X_std)  # Std
    n_components_std = pca_std.explained_variance_ratio_.size
    for i in range(1, n_components_std):
        expl_var = pca_std.explained_variance_ratio_.sum()
        if(expl_var > 0.9):
            n_components_std = i
            pca_std = PCA(n_components=n_components_std)
            pca_std.fit(X_std)
            break
    spectra_std_pca = pca_std.transform(X_std)
    pca_pwr = PCA()
    pca_pwr.fit(X_pwr)  # Pwr
    n_components_pwr = pca_pwr.explained_variance_ratio_.size
    for i in range(1, n_components_pwr):
        expl_var = pca_pwr.explained_variance_ratio_.sum()
        if(expl_var > 0.9):
            n_components_pwr = i
            pca_pwr = PCA(n_components=n_components_pwr)
            pca_pwr.fit(X_pwr)
            break
    spectra_pwr_pca = pca_pwr.transform(X_pwr)

    pca_norm = PCA()
    pca_norm.fit(X_norm)  # Norm
    n_components_norm = pca_norm.explained_variance_ratio_.size
    for i in range(1, n_components_norm):
        expl_var = pca_norm.explained_variance_ratio_.sum()
        if(expl_var > 0.9):
            n_components_norm = i
            pca_norm = PCA(n_components=n_components_norm)
            pca_norm.fit(X_norm)
            break
    spectra_norm_pca = pca_norm.transform(X_norm)

    # Create regressors
    print("Creating regressors...")
    dtrees = DecisionTreeRegressor()
    knn = KNeighborsRegressor()
    svr = svm.SVR()
    print("Testing regressors...")
    warnings.filterwarnings("ignore")
    results["DT_std_PCA"] = _doModelSelection(dtrees, spectra_std_pca, to_predict, multithread)
    results["KNN_std_PCA"] = _doModelSelection(knn, spectra_std_pca, to_predict, multithread)
    results["SVR_std_PCA"] = _doModelSelection(svr, spectra_std_pca, to_predict, multithread)
    results["DT_pwr_PCA"] = _doModelSelection(dtrees, spectra_pwr_pca, to_predict, multithread)
    results["KNN_pwr_PCA"] = _doModelSelection(knn, spectra_pwr_pca, to_predict, multithread)
    results["SVR_pwr_PCA"] = _doModelSelection(svr, spectra_pwr_pca, to_predict, multithread)
    results["DT_norm_PCA"] = _doModelSelection(dtrees, spectra_norm_pca, to_predict, multithread)
    results["KNN_norm_PCA"] = _doModelSelection(knn, spectra_norm_pca, to_predict, multithread)
    results["SVR_norm_PCA"] = _doModelSelection(svr, spectra_norm_pca, to_predict, multithread)
    if(not dim_red_only):
        results["DT"] = _doModelSelection(dtrees, X_orig, to_predict, multithread)
        results["KNN"] = _doModelSelection(knn, X_orig, to_predict, multithread)
        results["SVR"] = _doModelSelection(svr, X_orig, to_predict, multithread)
        results["DT_std"] = _doModelSelection(dtrees, X_std, to_predict, multithread)
        results["KNN_std"] = _doModelSelection(knn, X_std, to_predict, multithread)
        results["SVR_std"] = _doModelSelection(svr, X_std, to_predict, multithread)
        results["DT_pwr"] = _doModelSelection(dtrees, X_pwr, to_predict, multithread)
        results["KNN_pwr"] = _doModelSelection(knn, X_pwr, to_predict, multithread)
        results["SVR_pwr"] = _doModelSelection(svr, X_pwr, to_predict, multithread)
        results["DT_norm"] = _doModelSelection(dtrees, X_norm, to_predict, multithread)
        results["KNN_norm"] = _doModelSelection(knn, X_norm, to_predict, multithread)
        results["SVR_norm"] = _doModelSelection(svr, X_norm, to_predict, multithread)

    # Plot resulting scores
    print("Plotting resuling scores of the RMSECV")
    plt.figure(fig_size=(16, 12))
    for name, result in results.items():
        score_index = 0
        plt.bar(name, result[score_index])
    plt.xlabel("Model")
    plt.ylabel("Score (RMSECV)")
    plt.show()
    # Evaluate all regresors
    chosen_reg = results["DT_std_PCA"]
    for name, result in results.items():
        if(abs(result[0]) < abs(chosen_reg[0])):
            if("std_PCA" in name):
                pca = pca_std
            elif("pwr_PCA" in name):
                pca = pca_pwr
            elif("norm_PCA" in name):
                pca = pca_norm
            else:
                pca = None
            chosen_reg = (result[0], result[1], {"name": name, "pca": pca})

    # Return model of best regressor with the results
    return chosen_reg


def _testClassifiers(spectra, to_predict, multithread, dim_red_only, sam_sz):
    '''

    Test multiple classifiers (Support Vector Machines, Decision Tree,
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
        chosen_classifier : ndarray
            A 3 columns array that contains information of the best classifier
            [
                    0 -> Score of the best classifier
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
        raise ValueError(f"Shape mismatch: value array of size ({spectra.shape[0]}) and ({to_predict.shape[0]}) don't match.")

    # Results dictionary
    results = {}

    # Perform normalization
    print("Performing normalization...")
    spectra_std = spectra.copy()
    spectra_pwr = spectra.copy()
    spectra_norm = spectra.copy()
    std_scaler = StandardScaler()
    pwr_trans = PowerTransformer()
    normalizer = Normalizer()
    std_scaler.fit(spectra_std.loc[:, 'intensity'].tolist())
    pwr_trans.fit(spectra_pwr.loc[:, 'intensity'].tolist())
    normalizer.fit(spectra_norm.loc[:, 'intensity'].tolist())
    
    for i in spectra.index:
        new_sig = std_scaler.transform(spectra_std.at[i, 'intensity'].reshape(1,-1))
        spectra_std.at[i, 'intensity'] = new_sig[0]
        new_sig = pwr_trans.transform(spectra_pwr.at[i, 'intensity'].reshape(1,-1))
        spectra_pwr.at[i, 'intensity'] = new_sig[0]
        new_sig = normalizer.transform(spectra_norm.at[i, 'intensity'].reshape(1,-1))
        spectra_norm.at[i, 'intensity'] = new_sig[0]

    # Stack data in order to comply with scikit inputs
    X_orig = np.stack(spectra.loc[:, 'intensity'])
    X_std = np.stack(spectra_std.loc[:, 'intensity'])
    X_pwr = np.stack(spectra_pwr.loc[:, 'intensity'])
    X_norm = np.stack(spectra_norm.loc[:, 'intensity'])
    print("Shapes", X_orig.shape, X_std.shape, X_pwr.shape, X_norm.shape)

    # Perform dimensionality reduction
    print("Performing dimensionality reduction...")
    pca_std = PCA()
    pca_std.fit(X_std)  # Std
    n_components_std = pca_std.explained_variance_ratio_.size
    for i in range(1, n_components_std):
        expl_var = pca_std.explained_variance_ratio_.sum()
        if(expl_var > 0.9):
            n_components_std = i
            pca_std = PCA(n_components=n_components_std)
            pca_std.fit(X_std)
            break
    spectra_std_pca = pca_std.transform(X_std)
    pca_pwr = PCA()
    pca_pwr.fit(X_pwr)  # Pwr
    n_components_pwr = pca_pwr.explained_variance_ratio_.size
    for i in range(1, n_components_pwr):
        expl_var = pca_pwr.explained_variance_ratio_.sum()
        if(expl_var > 0.9):
            n_components_pwr = i
            pca_pwr = PCA(n_components=n_components_pwr)
            pca_pwr.fit(X_pwr)
            break
    spectra_pwr_pca = pca_pwr.transform(X_pwr)

    pca_norm = PCA()
    pca_norm.fit(X_norm)  # Norm
    n_components_norm = pca_norm.explained_variance_ratio_.size
    for i in range(1, n_components_norm):
        expl_var = pca_norm.explained_variance_ratio_.sum()
        if(expl_var > 0.9):
            n_components_norm = i
            pca_norm = PCA(n_components=n_components_norm)
            pca_norm.fit(X_norm)
            break
    spectra_norm_pca = pca_norm.transform(X_norm)

    # Create classifiers
    print("Creating classifiers...")
    dtrees = DecisionTreeClassifier()
    svc = svm.SVC()
    knn = KNeighborsClassifier()
    print("Testing classifiers...")
    warnings.filterwarnings("ignore")
    results["DT_std_PCA"] = _doModelSelection(dtrees, spectra_std_pca, to_predict, multithread)
    results["SVC_std_PCA"] = _doModelSelection(svc, spectra_std_pca, to_predict, multithread)
    results["KNN_std_PCA"] = _doModelSelection(knn, spectra_std_pca, to_predict, multithread)
    results["DT_pwr_PCA"] = _doModelSelection(dtrees, spectra_pwr_pca, to_predict, multithread)
    results["SVC_pwr_PCA"] = _doModelSelection(svc, spectra_pwr_pca, to_predict, multithread)
    results["KNN_pwr_PCA"] = _doModelSelection(knn, spectra_pwr_pca, to_predict, multithread)
    results["DT_norm_PCA"] = _doModelSelection(dtrees, spectra_norm_pca, to_predict, multithread)
    results["SVC_norm_PCA"] = _doModelSelection(svc, spectra_norm_pca, to_predict, multithread)
    results["KNN_norm_PCA"] = _doModelSelection(dtrees, spectra_norm_pca, to_predict, multithread)
    if(not dim_red_only):
        results["DT"] = _doModelSelection(dtrees, X_orig, to_predict, multithread)
        results["SVC"] = _doModelSelection(svc, X_orig, to_predict, multithread)
        results["KNN"] = _doModelSelection(knn, X_orig, to_predict, multithread)
        results["DT_std"] = _doModelSelection(dtrees, X_std, to_predict, multithread)
        results["SVC_std"] = _doModelSelection(svc, X_std, to_predict, multithread)
        results["KNN_std"] = _doModelSelection(knn, X_std, to_predict, multithread)
        results["DT_pwr"] = _doModelSelection(dtrees, X_pwr, to_predict, multithread)
        results["SVC_pwr"] = _doModelSelection(svc, X_pwr, to_predict, multithread)
        results["KNN_pwr"] = _doModelSelection(knn, X_pwr, to_predict, multithread)
        results["DT_norm"] = _doModelSelection(dtrees, X_norm, to_predict, multithread)
        results["SVC_norm"] = _doModelSelection(svc, X_norm, to_predict, multithread)
        results["KNN_norm"] = _doModelSelection(knn, X_norm, to_predict, multithread)

    # Evaluate all classifiers
    chosen_clf = results["DT_std_PCA"]
    for name, result in results.items():
        if(abs(result[0]) > abs(chosen_clf[0])):
            if("std_PCA" in name):
                pca = pca_std
            elif("pwr_PCA" in name):
                pca = pca_pwr
            elif("norm_PCA" in name):
                pca = pca_norm
            else:
                pca = None
            chosen_clf = (result[0], result[1], {"name": name, "pca": pca})

    # Return model of best classifiers with the results
    return chosen_clf


def _doModelSelection(estimator, X, Y, multithread):
    n_jobs = -1 if(multithread) else None
    print("Evaluating...")
    if(isinstance(estimator, DecisionTreeRegressor)):
        estimator = DecisionTreeRegressor()
        score_metrics = ["r2", "neg_mean_squared_error"]
        refit = "neg_mean_squared_error"
        params = {
                'criterion': ["mse", "friedman_mse", "mae"],
                'max_depth': np.linspace(1, 33, 3, endpoint=True),
                'min_samples_split': np.linspace(0.1, 1.0, 3, endpoint=True),
                'min_samples_leaf': np.linspace(0.1, 0.5, 3, endpoint=True)
                }
    elif(isinstance(estimator, KNeighborsRegressor) or isinstance(estimator,
         KNeighborsClassifier)):
        if(isinstance(estimator, KNeighborsRegressor)):
            estimator = KNeighborsRegressor(n_jobs=-1)
            score_metrics = ["r2", "neg_mean_squared_error"]
            refit = "neg_mean_squared_error"
        else:
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
    elif(isinstance(estimator, svm.SVR)):
        estimator = svm.SVR(gamma='scale')
        score_metrics = ["r2", "neg_mean_squared_error"]
        refit = "neg_mean_squared_error"
        params = {
                'C': np.linspace(1, 1e8, 8, endpoint=True, dtype=np.int8),
                'kernel': ['linear'],
                'epsilon': np.linspace(0.1, 1.0, 10, endpoint=True)
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
    clf = GridSearchCV(estimator, params, cv=5, error_score=np.nan,
                       scoring=score_metrics, refit=refit, n_jobs=n_jobs)
    clf.fit(X, Y)
    print("Done evaluating.")
    return clf.best_score_, clf.best_estimator_


def _trainModel(spectra, to_predict):
    model = spectra._model[1]
    model_info = spectra._model[2]
    data = np.stack(spectra.loc[:, 'intensity'])
    if("PCA" in model_info["name"]):
        pca = model_info["pca"]
        data = pca.transform(data)
    X, X_test, Y, Y_test = train_test_split(data, to_predict, test_size=0.15,
                                            shuffle=True, random_state=7)
    model.fit(X, Y)
    return {"Y_train": Y,
            "Y_test": Y_test,
            "Y_test_pred": model.predict(X_test),
            "Y_train_pred": model.predict(X)}


def _predict(spectra, index):
    model = spectra._model[1]
    model_info = spectra._model[2]

    if(index != -1 and not isinstance(index, int)):
        data = np.stack(spectra.loc[index, 'intensity'].copy())
    elif(index == -1):
        data = np.stack(spectra.loc[:, 'intensity'].copy())
    elif(isinstance(index, int)):
        data = np.stack(spectra.loc[index, 'intensity'].copy()).reshape(1, -1)

    if("PCA" in model_info["name"]):
        pca = model_info["pca"]
        data = pca.transform(data)

    return model.predict(data)
