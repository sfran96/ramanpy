 # -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:37:19 2020

@author: Francis Santos
"""
import numpy as np
import warnings
import matplotlib.pyplot as plt
import math
from pandas import DataFrame
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, Normalizer, label_binarize
from sklearn.cross_decomposition.pls_ import _PLS
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from scipy.stats import linregress
from seaborn import heatmap


class PLSReg(_PLS):
    def __init__(self, n_components=2, *, scale=False,
                 max_iter=500, tol=1e-06, copy=True, algorithm="nipals"):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="regression", mode="A",
            norm_y_weights=False, max_iter=max_iter, tol=tol,
            copy=copy, algorithm=algorithm)


def _testRegressors(spectra, to_predict, multithread, **kwargs):
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

    # Params
    if(multithread):
        n_jobs = -1
    else:
        n_jobs = None

    # Create initial pipeline
    pipe = Pipeline([
        ('scale', 'passthrough'),
        ('estimate', 'passthrough')
    ])

    # Create parameters
    param_grid = [
        {
            'scale': [Normalizer()],
            'estimate': [PLSReg(max_iter=1e4)],
            'estimate__n_components': np.linspace(1, 40, 40, endpoint=True, dtype=np.int8)
        },
        {
            'scale': [Normalizer()],
            'estimate': [Ridge(max_iter=1e4)],
            'estimate__alpha': np.logspace(-1, 3, 100)
        }
    ]

    # Test the grid of parameters
    grid = GridSearchCV(pipe, param_grid, cv=10, error_score=np.nan,
                    scoring="neg_mean_squared_error", refit=True, n_jobs=n_jobs,
                    verbose=10, **kwargs)
    grid.fit(spectra.intensity, to_predict)

    # Select best and return
    cv_results = DataFrame(grid.cv_results_)

    # Returns
    return _doModelSelection(cv_results)


def _testClassifiers(spectra, to_predict, multithread, **kwargs):    # Check types are correct and arrays are the same size
    if("Spectra" not in spectra.__class__.__name__):
        raise AttributeError("The spectra attribute must be of Spectra type.")
    if(not isinstance(to_predict, np.ndarray)):
        raise AttributeError("The to_predict attribute must be a numpy array.")
    if(spectra.shape[0] != to_predict.shape[0]):
        raise ValueError(f"Shape mismatch: value array of size \
                         ({spectra.shape[0]}) and ({to_predict.shape[0]}) \
                         don't match.")

    # Params
    if(multithread):
        n_jobs = -1
    else:
        n_jobs = None

    # Create initial pipeline
    pipe = Pipeline([
        ('scale', 'passthrough'),
        ('estimate', 'passthrough')
    ])

    # Create parameters
    param_grid = [
        {
            'scale': [Normalizer()],
            'estimate': [LinearSVC(max_iter=1e4, random_state=7)],
            'estimate__C': np.logspace(-2, 4, 50)
        },
        {
            'scale': [Normalizer()],
            'estimate': [KNeighborsClassifier(n_jobs=n_jobs)],
            'estimate__n_neighbors': np.arange(3, 10, 1).astype(int),
            'estimate__metric': ["euclidean", "minkowski", "chebyshev"]
        }
    ]

    X, y = spectra.intensity, label_binarize(to_predict, classes=np.unique(to_predict))

    # Test the grid of parameters
    grid = GridSearchCV(pipe, param_grid, cv=10, error_score=np.nan,
                    scoring="f1_weighted", refit=True, n_jobs=n_jobs,
                    verbose=10, **kwargs)

    # Fit
    grid.fit(X, y)

    # Select best and return
    cv_results = DataFrame(grid.cv_results_)

    # Returns
    return _doModelSelection(cv_results)


def _doModelSelection(cv_results):
    # Retrieve information of the highest rank estimator
    best_info = cv_results[cv_results.rank_test_score == 1].iloc[0]
    best = best_info.param_estimate
    # Check which type of estimator it is and make an instance
    if(isinstance(best, Ridge)):
        best.set_params(**{
            "alpha": best_info.param_estimate__alpha
            })
        rng = (0, 40)
        info = {"name": "Ridge", "type": "regression", "scaler": best_info.param_scale}
    elif(isinstance(best, PLSReg)):
        best.set_params(**{
            "n_components": best_info.param_estimate__n_components
            })
        rng = (40, 140)
        info = {"name": "PLS", "type": "regression", "scaler": best_info.param_scale}
    elif(isinstance(best, LinearSVC)):
        best.set_params(**{
            "C": best_info.param_estimate__C
            })
        rng = (0, 50)
        info = {"name": "SVC", "type": "classification", "scaler": best_info.param_scale}
    elif(isinstance(best, KNeighborsClassifier)):
        best.set_params(**{
            "n_neighbors": best_info.param_estimate__n_neighbors,
            "metric": best_info.param_estimate__metric
            })
        rng = (50, 71)
        info = {"name": "KNC", "type": "classification", "scaler": best_info.param_scale}
    # Plot results for each type of estimator
    if(info["type"] == "regression"):
        results_metrics_means = np.array((
            cv_results.iloc[:40].mean_test_score.mean(),
            cv_results.iloc[40:140].mean_test_score.mean()
            ))
        plt.figure(figsize=(12,8))
        plt.bar(["Ridge", "PLS"], results_metrics_means)
        plt.xlabel("Name")
        plt.ylabel("Mean MSE")
        plt.title("Mean MSE value across all the parameters tested on the different models")
        plt.show()
    elif(info["type"] == "classification"):
        results_metrics_means = np.array((
            cv_results.iloc[:50].mean_test_score.mean(),
            cv_results.iloc[50:71].mean_test_score.mean()
            ))
        plt.figure(figsize=(12,8))
        plt.bar(["SVC", "KNN"], results_metrics_means)
        plt.xlabel("Name")
        plt.ylabel("Mean weighted F1")
        plt.title("Mean weighted F1-value across all the parameters tested on the different models")
        plt.show()

    # Return results as [mean, estimator, estimator_info_dict]
    return cv_results[rng[0]:rng[1]].mean_test_score.mean(), best, info


def _trainModel(spectra, to_predict, show_graph=False, cv=10):
    # Obtain information from model structure
    model = spectra._model[1]
    model_info = spectra._model[2]
    data = spectra.intensity

    if(model_info["type"] == "regression"):
        X, y = data, to_predict
    elif(model_info["type"] == "classification"):
        X, y = data, label_binarize(to_predict, classes=np.unique(to_predict))

    # Scale
    scaler = model_info["scaler"]
    X = scaler.fit_transform(X)
    spectra._model[2]["scaler"] = scaler
    
    # Fit to model    
    model.fit(X, y)    

    # Predict train and test sets
    y_train_pred = model.predict(X)

    if(model_info["type"] == "regression"):
        y_train_pred = y_train_pred.reshape(-1)

    results = {"Y_train": y,
            "Y_train_pred": y_train_pred,
            "X_train": X
            }

    # Plot results
    if(show_graph):
        _plotResults(results, model_info["type"], model, spectra, cv)

    return results


def _predict(spectra, index):
    model = spectra._model[1]
    model_info = spectra._model[2]

    if(isinstance(index, (list, tuple, np.ndarray))):
        data = spectra.intensity[index].copy()
    elif(isinstance(index, int) and index != -1):
        data = spectra.intensity[index].copy().reshape(1, -1)
    elif(index == -1):
        data = spectra.intensity.copy()

    scaler = model_info["scaler"]
    data = scaler.transform(data)

    return model.predict(data)


def _plotResults(results, estimator_type, model, spectra, cv):
    plt.figure(figsize=(15,20))
    if(estimator_type == "regression"):        
        regress = linregress(results["Y_train"], results["Y_train_pred"])
        print(f"Metrics: \nRMSECV = {round(math.sqrt(abs(cross_val_score(model, results['X_train'], results['Y_train'], scoring='neg_mean_squared_error', cv = cv).mean())), 2)} \nR2 = {round(r2_score(results['Y_train'], results['Y_train_pred']), 2)} \nR coefficient = {round(regress.rvalue, 2)}")
        # print(f"Testing dataset metrics: \nRMSEP = {round(math.sqrt(abs(mean_squared_error(results['Y_test'], results['Y_test_pred']))), 2)} \nR2 = {round(r2_score(results['Y_test'], results['Y_test_pred']), 2)}\n")

        if(isinstance(model, _PLS)):
            n_rows = 4   
        else:
            n_rows = 3
        # Position 1-1
        ax1 = plt.subplot(n_rows, 2, 1)
        plt.scatter(results["Y_train"], results["Y_train_pred"], color="gray")
        # plt.scatter(results["Y_test"], results["Y_test_pred"], color="black", label="Testing set")
        ax1.set_title("Predicted vs. Measured")
        ax1.set_xlabel("Measured Value")
        ax1.set_ylabel("Predicted Value")
        plt.plot(results["Y_train"], (results["Y_train"]*regress.slope + regress.intercept), label="Model's regression line")
        min_value = np.min([results["Y_train"].min(), results["Y_train_pred"].min()])
        max_value = np.max([results["Y_train"].max(), results["Y_train_pred"].max()])
        plt.plot(np.arange(min_value*0.7, max_value*1.2, 0.1), np.arange(min_value*0.7, max_value*1.2, 0.1), label="Ideal regression line", color="black")
        ax1.set_xlim([min_value*0.8, max_value*1.1])
        ax1.set_ylim([min_value*0.8, max_value*1.1])
        ax1.legend()
        # Positoin 1-2
        ax2 = plt.subplot(n_rows, 2, 2)
        plt.scatter(results["Y_train_pred"], results["Y_train"] - results["Y_train_pred"], color="gray")
        # plt.scatter(results["Y_test"], results["Y_test"] - results["Y_test_pred"], color="black", label="Testing set")
        ax2.set_title("Residuals Plot")
        ax2.set_xlabel("Predicted Value")
        ax2.set_ylabel("Residuals")
        # ax2.legend()
        # Position 2-1 and 2-2
        ax3 = plt.subplot(n_rows, 1, 2)
        weights = model.coef_
        plt.plot(spectra.wavenumbers, weights)
        ax3.set_title("Weights of the shifts on the model")
        ax3.set_xlabel("Raman shift ($cm^{-1}$)")
        ax3.set_ylabel("Weight")
        ax3.set_xlim([spectra.wavenumbers.min(), spectra.wavenumbers.max()])
        # Position 4-1 and 4-2
        ax5 = plt.subplot(n_rows, 1, 3)
        plt.hist(results["Y_train"] - results["Y_train_pred"], 10, density=True, alpha=0.5)
        # plt.hist(results["Y_test"] - results["Y_test_pred"], 10, density=True, alpha=0.75, label="Testing set")
        ax5.set_title("Residuals distribution plot")
        ax5.set_xlabel("Residuals")
        ax5.set_ylabel("Ocurrences (w.r.t. area = 1)")
        # ax5.legend()
        # Position 3-1 and 3-2
        if(isinstance(model, _PLS)):
            ax4 = plt.subplot(n_rows, 1, 4)
            x_weights = model.x_weights_
            plt.plot(spectra.wavenumbers, x_weights[:, 0], label="PC1")
            if(x_weights.shape[1] > 1):
                plt.plot(spectra.wavenumbers, x_weights[:, 1], label="PC2")
            ax4.set_title("Weights of the shifts on the model of the PCs")
            ax4.set_xlabel("Raman shift ($cm^{-1}$)")
            ax4.set_ylabel("Weight")
            ax4.set_xlim([spectra.wavenumbers.min(), spectra.wavenumbers.max()])
            ax4.legend()
    elif(estimator_type == "classification"):
        scoring = "f1_weighted"
        classes = np.unique(results["Y_train"])
        n_classes = classes.shape[0]
        # ROC Curve
        ax1 = plt.subplot(2, 1, 1)
        # # Compute ROC curve and ROC area for each class
        y_test = results["Y_train"]
        y_score = model.decision_function(results["X_train"]) if isinstance(model, LinearSVC) else model.predict_proba(results["X_train"])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if (n_classes > 2):
            for i in range(n_classes):            
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], lw=2,
                        label='ROC curve of class {0} (area = {1:0.2f})'
                        ''.format(classes[i], roc_auc[i]))  
        else:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                    label='ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_title("ROC Curve")
        ax1.set_xlabel("False Positives Rate")
        ax1.set_ylabel("True Positives Rate")
        ax1.legend()
        # Confusion Matrix
        ax2 = plt.subplot(2, 2, 3)
        heatmap(confusion_matrix(results["Y_train"], results["Y_train_pred"]),
                                 xticklabels=classes, yticklabels=classes,
                                 cmap="Greens",
                                 annot=True,
                                 fmt="g")
        ax2.set_title("Confusion matrix for training dataset")
        ax2.set_ylim([0, n_classes])
        # Classification report
        ax4 = plt.subplot(2, 2, 4)
        p = cross_val_score(model, results['X_train'], results['Y_train'], scoring='precision', cv = cv)
        r = cross_val_score(model, results['X_train'], results['Y_train'], scoring='recall', cv = cv)
        f1 = cross_val_score(model, results['X_train'], results['Y_train'], scoring='f1_weighted', cv = cv)
        a = cross_val_score(model, results['X_train'], results['Y_train'], scoring='accuracy', cv = cv)

        r1 = [0]
        r2 = [x + 0.25 for x in r1]
        r3 = [x + 0.25 for x in r2]
        r4 = [x + 0.25 for x in r3]
        plt.bar(r1, [p], width=0.25, label="Precision", color="black")
        plt.bar(r2, [r], width=0.25, label="Recall", color="gray")
        plt.bar(r3, [f1], width=0.25, label="F1-Score", color="darkgray")
        plt.bar(r4, [a], width=0.25, label="Accuracy", color="lightgray")
        ax4.set_title("Classification CV summary report")
        ax4.set_xlabel("Score")
        ax4.set_ylabel("Value")
        ax4.legend(loc="lower center")

def _resultsIsolatedFrequencies(spectra, to_predict, slots=10, cv=10):
    # Get spectra for the different ranges
    if(isinstance(slots, int)):
        splitted_wavenumbers_indices = np.array_split(range(len(spectra.wavenumbers)), slots)
    else:
        AttributeError("The slots property must be an integer.")
    # Train already existing model to the different ranges
    model = spectra._model
    model_info = model[2]
    results = {}
    for wavenumbers_indices in splitted_wavenumbers_indices:
        spectra_aux = spectra.iloc[:, wavenumbers_indices]
        spectra_aux._model = model
        key = "{}-{}$cm^{{-1}}$".format(spectra.wavenumbers[wavenumbers_indices[0]], spectra.wavenumbers[wavenumbers_indices[-1]])
        results[key] = spectra_aux.trainModel(to_predict, False)
    
    if(model_info["type"] == "regression"):
        plt.figure(figsize=(15,28))
        plt.subplot(211)
        # For each results dictionary, calculate the cross-validation r2-score using Python's comprehensions
        r2scores = [cross_val_score(model[1], r["X_train"], r["Y_train"], scoring="r2", cv = cv).mean() for r in results.values()]
        clrs = ['green' if (x in np.sort(r2scores)[-3:]) else 'grey' for x in r2scores]
        plt.bar(results.keys(), r2scores, color=clrs) 
        plt.xticks(rotation=90)
        plt.ylabel("R2 score")
        plt.xlabel("Raman shift range")
        plt.title("R2 CV of the different wavenumber groups (higher is better)")

        plt.subplot(212)
        # For each results dictionary, calculate the cross-validation RMSECV using Python's comprehensions
        rmses = [math.sqrt(abs(cross_val_score(model[1], r["X_train"], r["Y_train"], scoring="neg_mean_squared_error", cv = cv).mean())) for r in results.values()]
        clrs = ['green' if (x in np.sort(rmses)[:3]) else 'grey' for x in rmses]
        plt.bar(results.keys(), rmses, color=clrs) 
        plt.xticks(rotation=90)
        plt.ylabel("RMSECV")
        plt.xlabel("Raman shift range")
        plt.title("RMSE CV of the different wavenumber groups (smaller is better)")
    elif(model_info["type"] == "classification"):        
        plt.figure(figsize=(15,60))
        plt.subplot(411)
        # For each results dictionary, calculate the cross-validation r2-score using Python's comprehensions
        weighted_f1s = [cross_val_score(model[1], r["X_train"], r["Y_train"], scoring="f1_weighted", cv = cv).mean() for r in results.values()]
        clrs = ['green' if (x in np.sort(weighted_f1s)[-3:]) else 'grey' for x in weighted_f1s]
        plt.bar(results.keys(), weighted_f1s, color=clrs) 
        plt.xticks(rotation=90)
        plt.ylabel("Weighted F1 score")
        plt.xlabel("Raman shift range")
        plt.title("Weighted F1 CV of the different wavenumber groups (higher is better)")

        plt.subplot(412)
        # For each results dictionary, calculate the cross-validation RMSECV using Python's comprehensions
        accuracies = [math.sqrt(abs(cross_val_score(model[1], r["X_train"], r["Y_train"], scoring="accuracy", cv = cv).mean())) for r in results.values()]
        clrs = ['green' if (x in np.sort(accuracies)[-3:]) else 'grey' for x in accuracies]
        plt.bar(results.keys(), accuracies, color=clrs) 
        plt.xticks(rotation=90)
        plt.ylabel("Accuracy")
        plt.xlabel("Raman shift range")
        plt.title("Accuracy CV of the different wavenumber groups (higher is better)")

        plt.subplot(413)
        # For each results dictionary, calculate the cross-validation RMSECV using Python's comprehensions
        precisions = [math.sqrt(abs(cross_val_score(model[1], r["X_train"], r["Y_train"], scoring="precision", cv = cv).mean())) for r in results.values()]
        clrs = ['green' if (x in np.sort(precisions)[-3:]) else 'grey' for x in precisions]
        plt.bar(results.keys(), precisions, color=clrs) 
        plt.xticks(rotation=90)
        plt.ylabel("Precision")
        plt.xlabel("Raman shift range")
        plt.title("Precision CV of the different wavenumber groups (higher is better)")

        plt.subplot(414)
        # For each results dictionary, calculate the cross-validation RMSECV using Python's comprehensions
        recalls = [math.sqrt(abs(cross_val_score(model[1], r["X_train"], r["Y_train"], scoring="recall", cv = cv).mean())) for r in results.values()]
        clrs = ['green' if (x in np.sort(recalls)[-3:]) else 'grey' for x in recalls]
        plt.bar(results.keys(), recalls, color=clrs) 
        plt.xticks(rotation=90)
        plt.ylabel("Recall")
        plt.xlabel("Raman shift range")
        plt.title("Recall CV of the different wavenumber groups (higher is better)")