## IMPORT STATEMENTS
## NOT ALL NEEDED, USE AS NECESSARY

import pandas as pd
import numpy as np
import re
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

## READ DATAFRAME AND CONVERT ALL TO LOWERCASE

df = pd.read_excel(open('Referral Decisions.xlsx', 'rb'), sheet_name='Combined')
df = df.dropna(subset=['Risk Details'])
df = df.dropna(subset=['Reason(s) for Decision'])
df['Risk Details'] = df['Risk Details'].apply(lambda x:str(x).lower())
df['Reason(s) for Decision'] = df['Reason(s) for Decision'].apply(lambda x:str(x).lower())

## SUBSET BY RISK ATTRIBUTE

brushDF = df[df['Risk Details'].str.contains('brush')]
responseDF = df[df['Risk Details'].str.contains('response')]
lossDF = df[df['Risk Details'].str.contains('loss')]
alarmDF = df[df['Risk Details'].str.contains('alarm')]

## MAKE NEW COLUMNS FOR EACH ATTRIBUTE

df['Not In Brush'] = brushDF['Risk Details'].str.contains('not in brush') ## MAY BE CONFUSING TRUE/FALSE VALUE - WILL BE SWITCHED TO MAKE MORE INUTITIVE SENSE LATER
df['No Losses'] = lossDF['Risk Details'].str.contains('no loss')
df['No Alarms'] = alarmDF['Risk Details'].str.contains('no alarm')

## IMPUTE MISSING VALUES BASED ON PROBABILITY DISTRIBUTION
## OK TO DO FOR CATEGORICAL VARIABLES (TRUE/FALSE)

s = df['Not In Brush'].value_counts(normalize=True) ## MAY NOT BE COMPELTELY SOUND STATISTICALLY, BUT WILL ALLOW A MODEL TO BE FIT
missing = df['Not In Brush'].isnull()
df.loc[missing,'Not In Brush'] = np.random.choice(s.index, size=len(df[missing]),p=s.values)

s2 = df['No Losses'].value_counts(normalize=True)
missing2 = df['No Losses'].isnull()
df.loc[missing2, 'No Losses'] = np.random.choice(s2.index, size=len(df[missing2]), p=s2.values)

s3 = df['No Alarms'].value_counts(normalize=True)
missing3 = df['No Alarms'].isnull()
df.loc[missing3, 'No Alarms'] = np.random.choice(s3.index, size=len(df[missing3]), p=s3.values)

## CHANGE COLUMN TYPES TO BOOLEAN

df['Not In Brush'] = df['Not In Brush'].astype('bool')
df['No Alarms'] = df['No Alarms'].astype('bool')
df['No Losses'] = df['No Losses'].astype('bool')
# df['Not In Brush'].dtypes
# df['No Losses'].dtypes
# df['No Alarms'].dtypes

## INVERT COLUMN TRUE/FALSE VALUES TO MAKE MORE INUITIVE SENSE WHEN INCORPORATED INTO DECISION TREE

df['Alarms'] = ~df['No Alarms']
df = df.drop(['No Alarms'], axis=1)

df['In Brush'] = ~df['Not In Brush']
df = df.drop(['Not In Brush'], axis=1)

df['Previous Losses'] = ~df['No Losses']
df = df.drop(['No Losses'], axis = 1)

## ADD IN YEAR BUILT AND RESPONSE TIME COLUMNS

## YEAR BUILT

p = re.compile('[1][7-9][0-9]{2}')
def get_years(x):
    if len(p.findall(x)) >= 1:
        return p.findall(x)[0]
df['Year Built'] = df['Risk Details'].apply(get_years)

# RESPONSE TIMES

rt1 = re.compile('(\d{2}).{3,15}response time')
rt2 = re.compile('response time.{3,15}(\d{2})')
def get_response_times1(x):
    if len(rt1.findall(x)) == 1:
        return rt1.findall(x)
def get_response_times2(x):
    if len(rt2.findall(x)) == 1:
        return rt2.findall(x)

response_times1 = responseDF['Risk Details'].apply(get_response_times1)
response_times2 = responseDF['Risk Details'].apply(get_response_times2)

df['Response Time'] = response_times1
df['Response Time'] = df['Response Time'].combine_first(response_times2)

## See count for each risk attribute

# print(len(df['Not In Brush'].dropna()))
# print(len(df['No Losses'].dropna()))
# print(len(df['No Alarms'].dropna()))
# print(len(df['Year Built'].dropna()))
# print(len(df['Response Time'].dropna()))

## YEAR BUILT IMPUTATION

df['Year Built'] = df['Year Built'].fillna(0)
df['Year Built'] = df['Year Built'].astype(int)
df['After 1950'] = df['Year Built'] ## CONVERT NUMERICAL ATTRIBUTE TO CATEGORICAL CONDITIONAL

def after_1950(x):
    if x == 0:
        return None
    elif x >= 1950:
        return 'True'
    else:
        return 'False'

df['After 1950'] = df['After 1950'].apply(after_1950)

s4 = df['After 1950'].value_counts(normalize=True)
missing4 = df['After 1950'].isnull()
df.loc[missing4, 'After 1950'] = np.random.choice(s4.index, size=len(df[missing4]), p=s4.values)

def change_bool(x):
    if x == 'True':
        return True
    else:
        return False
df['After 1950'] = df['After 1950'].apply(change_bool)

## RESPONSE TIME IMPUTATION

dfResponse = df.dropna(subset=['Response Time']) ## SAME APPROACH AS YEAR BUILT IMPUTATION

def get_num(x):
    return int(x[0])

dfResponse['Response Time'] = dfResponse['Response Time'].apply(get_num)
df['Response Time'] = dfResponse['Response Time']

def less_20(x):
    if x <= 20.0:
        return 'True'
    elif x > 20.0:
        return 'False'
    else:
        return None

df['Response Time < 20 min'] = df['Response Time'].apply(less_20)

s5 = df['Response Time < 20 min'].value_counts(normalize=True)
missing5 = df['Response Time < 20 min'].isnull()
df.loc[missing5, 'Response Time < 20 min'] = np.random.choice(s4.index, size=len(df[missing5]), p=s5.values)

df['Response Time < 20 min'] = df['Response Time < 20 min'].apply(change_bool)

## ADD APPROVAL COLUMN

df['Approval'] = df['Declined?']

def switch_val(x):
    if x == 0:
        return 1
    else:
        return 0

df['Approval'] = df['Approval'].apply(switch_val)

## CLEAN DATA FOR DECISION TREE DATAFRAME

dt_df = df
dt_df = dt_df.drop(['Date', 'Agency Name', 'Office Location', 'Contact Name', 'Source (Call/Email)', 'Insured Last Name', 'Insured First Name', 'Product/Occupancy', 'City', 'State', 'Zip', 'Protection Class'], axis=1)
dt_df = dt_df.drop(['System Reason(s) for Referral', 'Risk Details', 'Reason(s) for Decision', 'MARKEL CONFIDENTIAL     Follow up & FYI Notes', 'Time Study - Minutes Spent', 'Underwriter', 'Flagged for Audit', 'Audit Notes'], axis=1)
dt_df = dt_df.drop(['Approve/Subj', 'Declined?', 'Year Built', 'Covg A', 'TIV', 'Response Time'], axis=1)

## TRAIN/TEST SPLIT

X = dt_df.loc[:, dt_df.columns != 'Approval']
y = dt_df['Approval']
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=40)

## RESAMPLE DATASET
## NEEDS TO BE DONE OR DECISION TREE WILL NOT PREDICT ANY OF THE UNDER REPRESENTED CATEGORY (DECLINES)

ros = RandomUnderSampler(random_state=10) ## OVERSAMPLE OR UNDERSAMPLE TO SUITE YOUR METRIC SATISFACTION
X_resampled, y_resampled = ros.fit_resample(train_X, train_y)

## TRAIN DECISION TREE

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_resampled, y_resampled)

# clf.get_depth() ## DEFAULT DEPTH VALUE FOR THIS TREE IS 5

## PREDICT

predicted_y = clf.predict(test_X)

## SEE CLASSIFICATION

print(classification_report(test_y, predicted_y))
tn, fp, fn, tp = confusion_matrix(test_y, predicted_y).ravel()
print(tn, fp, fn, tp)
print(confusion_matrix(test_y, predicted_y))

# print(len(y[y==0]))
# print(len(y[y==1]))

## SEE DECISION PATH OF THE TREE
## REQUIRES IMPORTS AND MAYBE PACKAGE INSTALLATION
## WILL GENERATE A .dot FILE WHICH YOU CAN COPY AND PASTE INTO http://webgraphviz.com/

export_graphviz(clf, out_file="decision_tree.dot", feature_names=X.columns)
with open("decision_tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


