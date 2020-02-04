## THIS PYTHON FILE TRANSFORMS THE BINDING E&S DATASET INTO A DOCUMENT TERM MATRIX
## A DOCUMENT TERM MATRIX USES A "BAG OF WORDS" STRATEGY TO REPRESENT EVERY SINGLE IMPORTANT WORD IN THE INPUT TEXT ON WHICH A MODEL CAN BE APPLIED
## IN THIS APPROACH, EVERY WORD BECOMES A FEATURE IN THE DATASET
## PURELY DONE FOR EDUCATION PURPOSES, LIKELY NOT APPLICABLE DUE TO LACK OF INTERPRETABILITY AND TOO COMPLEX

## IMPORT STATEMENTS
## NOT ALL NEEDED, USE AS NECESSARY

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.preprocessing import StandardScaler
from imblearn import FunctionSampler
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
import parfit.parfit as pf
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

## READ DATAFRAME AND COVERT ALL TO LOWER CASE

df = pd.read_excel(open('Referral Decisions.xlsx', 'rb'), sheet_name='Combined')
df = df.dropna(subset=['Risk Details'])
df = df.dropna(subset=['Reason(s) for Decision'])
df['Risk Details'] = df['Risk Details'].apply(lambda x:str(x).lower())
df['Reason(s) for Decision'] = df['Reason(s) for Decision'].apply(lambda x:str(x).lower())

## REMOVE ALL SPECIAL CHARCTERS AND NUMBERS
## LIKELY MORE EFFICIENT TO WRITE THESE REGEX FUNCTIONS BUT THIS DOES THE JOB

df['Risk Details'] = df['Risk Details'].map(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))
df['Risk Details'] = df['Risk Details'].map(lambda x: re.sub('[0-9]+', '', x))
df['Risk Details'] = df['Risk Details'].map(lambda x: re.sub('  +', ' ', x))
df['Risk Details'] = df['Risk Details'].map(lambda x: re.sub(' k +', ' ', x))
df['Risk Details'] = df['Risk Details'].map(lambda x: re.sub(r'\b[a-zA-Z]\b', '', x))
df['Risk Details'] = df['Risk Details'].map(lambda x: re.sub('  +', ' ', x))

## STEM WORDS
## WILL CONVERT WORDS TO THEIR ROOT WORD
## FLYING -> FLY

ps = PorterStemmer()
# wl = WordNetLemmatizer()
def stem_sentence(sentence):
    tokens = sentence.split()
    stem_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stem_tokens)

df['Risk Details'] = df['Risk Details'].apply(stem_sentence)

## CREATE DOCUMENT TERM MATRIX AND REMOVE STOP WORDS
## STOP WORDS ARE MEANINGLESS WORDS LIKE "a" OR "the"

docs = df['Risk Details'].tolist()
vec = TfidfVectorizer(stop_words='english')
X = vec.fit_transform(docs)
dtm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

## ADD APPROVAL COLUMN

df['Approval'] = df['Declined?']

def switch_val(x):
    if x == 0:
        return 1
    else:
        return 0

df['Approval'] = df['Approval'].apply(switch_val)

## TRAIN/TEST SPLIT

X = dtm.loc[:, dtm.columns != 'Approval']
y = dtm['Approval']
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)

## RESAMPLE DATASET

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(train_X, train_y)

## SGDClassifier GridSearch

grid = {
    #'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    #'l1_ratio': [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    #'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    #'class_weight': ['balanced']
}

paramGrid = ParameterGrid(grid)

bestModel, bestScore, allModels, allScores = pf.bestFit(SGDClassifier, paramGrid, X_resampled, y_resampled, test_X, test_y, metric=roc_auc_score, scoreLabel="AUC", n_jobs=1)

## FEATURE REDUCTION

pca = decomposition.PCA(n_components=50)
Xarray = np.array(X)
reduced = pca.fit(Xarray).transform(Xarray)

np.set_printoptions(precision=2,suppress=True)
print(reduced)

print(pca.explained_variance_ratio_)

## FIT NAIVE BAYES MODEL
## NAIVE BAYES MODELS TEND TO WORK THE BEST FOR DOCUMENT TERM MATRICES THAT HAVE HIGH DIMENSIONALITY

clf = MultinomialNB()
clf.fit(X_resampled, y_resampled)

## FIT STOCHASTIC GRADIENT DESCENT MODEL (SGD)

clf2 = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=1)
clf2 = clf2.fit(X_resampled, y_resampled)

## PREDICT

pred_y = clf.predict(test_X)

## CUTOFF VALUE/POINT
## NEED COST INFORMATION FOR FALSE POSITIVES AND FALSE NEGATIVES

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

threshold = Find_Optimal_Cutoff(test_y, pred_y)
print(threshold)

## CHECK CLASSIFICATION

print(accuracy_score(test_y, pred_y))
print(classification_report(test_y, pred_y))
print(confusion_matrix(test_y, pred_y))
