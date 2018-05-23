import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.ensemble import RandomForestClassifier

def plot_with_predict(X_values, y_values, model, y_label):
    y_predict = model.predict(X_values)

    plt.plot(y_values,y_predict, 'o', alpha = 0.5)
    plt.xlabel('actual ' + y_label)
    plt.ylabel('predicted ' + y_label)
    plt.grid()
    return

def plot_residuals(X, y_predict,y_actual,var_to_check):
    col_size = 3
    row_size = round(len(var_to_check)/col_size,0)+1
    fig=plt.figure(figsize=(10,2.5*row_size),dpi=100)
    for i, var in enumerate(var_to_check):
        ax=fig.add_subplot(row_size,col_size,i+1)
        # use the unscaled X against the y that is predicted from the scaled X - scaling doesn't affect output
        # also, plotting the scaled X values is not helpful in understanding the affect of X on y prediction
        ax.plot(X[var], y_predict - y_actual,'o', alpha=0.25)
        ax.set_title(var+" residuals (train)")
        ax.axhline(+1,color='r',lw=1,ls='-');
        ax.axhline(-1,color='r',lw=1,ls='-');
        ax.set_xlabel(var)
        ax.set_ylabel('prediction error')
        ax.grid()
    fig.tight_layout()
    return

def plot_feature_importance(X,y,model,X_col):
    forest_scores = np.array([est.score(X,y) for est in model.estimators_])
    forest_best = np.argmax(forest_scores)
    print(str(forest_best) + ", " + str(forest_scores[forest_best]))

    print('std = ' + str(np.std(forest_scores)))
    forest_importances = model.estimators_[forest_best].feature_importances_
    indices = np.argsort(forest_importances)

    # Plot the feature importances of the best tree in the forest
    plt.figure(figsize=(8,6),dpi=150)
    plt.title("Feature Importance")
    plt.barh(range(len(X_col)), forest_importances[indices])
    plt.yticks(range(len(X_col)), [X_col[i] for i in indices])
    plt.ylim([-1, len(X_col)]);
    
    return