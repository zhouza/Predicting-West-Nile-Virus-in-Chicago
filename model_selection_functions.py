import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from imblearn.over_sampling import SMOTE 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import *
from sklearn import metrics
from sklearn.preprocessing import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import *
from sklearn.dummy import DummyClassifier
import itertools
from sklearn.tree import export_graphviz
from sklearn.model_selection import *
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import *
from sklearn.svm import *
from sklearn.decomposition import PCA
import re
import pprint

def get_metrics_scores(y_true,y_predict):
    '''

    '''
    acc_score = round(metrics.accuracy_score(y_true,y_predict),4)
    preci_score = round(metrics.precision_score(y_true,y_predict),4)
    recall_score = round(metrics.recall_score(y_true,y_predict),4)
    f1_score = round(metrics.f1_score(y_true,y_predict),4)
    
    score_dict = {'accuracy': acc_score,
                  'precision': preci_score,
                  'recall': recall_score,
                 'f1': f1_score,
                 }
    
    return score_dict

def get_cv_scores(model,X,y_true,cv=5,scoring=['accuracy','precision','recall','f1']):
    '''

    '''
    score_dict = {}
    for metric in scoring:
        score = np.round(cross_val_score(model,X,y_true,cv=5,scoring=metric),4)
        score_dict[metric]=list(score)
    return score_dict

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix, both regular and normalized.
    """
    
    plt.figure(figsize=(15,8),dpi=140);
    fig, ax = plt.subplots(1,2)
    
    for normalize in [0,1]:
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            #print("Normalized confusion matrix")
        #else:
            #print('Confusion matrix, without normalization')

        #print(cm)
        
        img = ax[normalize].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        if normalize:
            ax[normalize].set_title('Normalized confusion matrix')
        else:
            ax[normalize].set_title('Confusion matrix')
        fig.colorbar(img, ax=ax[normalize])
        tick_marks = np.arange(len(classes))
        plt.sca(ax[normalize])
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax[normalize].text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.ylabel('True label');
        plt.xlabel('Predicted label');

    plt.tight_layout(w_pad=5,rect=(0,0,2,1));

def gen_resample(X,y,ratio):
    '''

    '''
    sm = SMOTE(ratio=ratio,random_state=42)
    X_rsmpl, y_rsmpl = sm.fit_sample(X, y)
    df_rsmpl = pd.DataFrame(np.concatenate((X_rsmpl,y_rsmpl.reshape(-1,1)),axis=1),columns=list(X.columns)+[y.name])   
    ratio_name = re.sub(r'[.]','',str(ratio))

    pickle.dump(X_rsmpl, open('pickles/X_rsmpl_train_'+ratio_name+'.pkl', 'wb'))
    pickle.dump(y_rsmpl,open('pickles/y_rsmpl_train_'+ratio_name+'.pkl','wb'))
    pickle.dump(df_rsmpl, open('pickles/df_rsmpl_train_'+ratio_name+'.pkl', 'wb'))
    
    return(X_rsmpl,y_rsmpl)

def gen_model_rsmpl(ratio,model,model_name):
    '''

    '''
    ratio_name = re.sub(r'[.]','',str(ratio))
    X_train = pickle.load(open('pickles/X_rsmpl_train_'+ratio_name+'.pkl','rb'))
    y_train = pickle.load(open('pickles/y_rsmpl_train_'+ratio_name+'.pkl','rb'))
    model.fit(X_train,y_train)
    pickle.dump(model,open('pickles/'+model_name+'_'+ratio_name+'.pkl','wb'))
    
    return(model)

def run_model_comparison(X,y_true,model_dict):
    '''
    models in model_list must already be fitted    
    '''
    for model in model_dict.keys():
        model_name = model
        y_pred = model_dict[model].predict(X)     
        scores = get_metrics_scores(y_true,y_pred)
        cv_scores = get_cv_scores(model_dict[model],X,y_true,5)
        print(model_name)
        print('training scores:')
        pprint.pprint(scores)
        print('cv scores:')
        pprint.pprint(cv_scores)
        print('\n')
        
        # Compute confusion matrix
        cf_matrix = metrics.confusion_matrix(y_true,y_pred)
        np.set_printoptions(precision=2)

        # Plot confusion matrixes
        plot_confusion_matrix(cf_matrix, classes=['Negative','Positive'])
        plt.show();

    return

def get_param_metrics(X,y,model):
    '''returns best params, training scores, cv_scores
        for a randomizedsearchcv or gridsearchcv
    '''
    pprint.pprint(model.best_params_)
    pprint.pprint(get_metrics_scores(y,model.best_estimator_.predict(X)))
    pprint.pprint(get_cv_scores(model.best_estimator_,X,y,cv=5))
    return