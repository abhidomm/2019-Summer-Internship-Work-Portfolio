## THIS PYTHON FILE WAS USED PURELY FOR EDA PURPOSES AND IS VERY MESSY
## WADE THROUGH THE SWAMP AT YOUR OWN RISK

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize

df = pd.read_excel(open('Referral Decisions.xlsx', 'rb'), sheet_name='Combined')
# dfCopy = df.copy()
# dfCopy = dfCopy.dropna(subset=['Risk Details'])
# print(len(dfCopy))
# print(len(df))
df = df.dropna(subset=['Risk Details'])
df = df.dropna(subset=['Reason(s) for Decision'])
df['Risk Details'] = df['Risk Details'].apply(lambda x:str(x).lower())
df['Reason(s) for Decision'] = df['Reason(s) for Decision'].apply(lambda x:str(x).lower())
#print(df['Risk Details'])

## EDA

App_Dec = sns.countplot(x="Declined?", data=df) ## Re-iterate the purpose/objective of this project - too many approved policies coming to underwriters' desks
App_Dec.set(xlabel = 'Approvals (0) and Declines (1)', ylabel = 'Frequency', title = 'Declined? Classification')
plt.show()

Diff_Dec = sns.countplot(x="Approve/Subj", data=df) ## Different decisons
Diff_Dec.set(xlabel= 'Different Types of Decisions', ylabel= 'Frequency', title = 'Approve/Subj Classification')
plt.show()

## Biggest risk factors in policies that DO NOT decline

approvalDF = df[df['Declined?'] == 0]

#print(approvalDF)

comment_words = ' '
stopwords = list(STOPWORDS) + ['insured', 'home', 'insd', 'will', 'property', 'agent', 'loss', 'dwelling', 'risk']

for val in approvalDF['Risk Details']:
    val = str(val)
    tokens = val.split()

    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    for words in tokens:
        comment_words = comment_words + words + ' '

def red_color_func(word, font_size, position, orientation, random_state=None):
    return "hsl(10, 100%%, %d%%)" % np.random.randint(40, 100)
wordcloud = WordCloud(width = 800, height = 800, background_color='firebrick', stopwords=stopwords).generate(comment_words)

plt.figure()
plt.imshow(wordcloud)
plt.show()

## Biggest risk factors in policies that DO decline

declineDF = df[df['Declined?'] == 1]

#print(declineDF)

comment_words = ' '
stopwords = list(STOPWORDS) + ['insured', 'home', 'insd', 'will', 'property', 'agent', 'updated', 'loss', 'dwelling']

for val in declineDF['Risk Details']:
    val = str(val)
    tokens = val.split()

    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    for words in tokens:
        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width = 800, height = 800, background_color='white',stopwords=stopwords).generate(comment_words)

plt.figure()
plt.imshow(wordcloud)
plt.show()

## Wordcloud not providing any meaningful info for reason for decision TEXT

brush_count_all = df['Risk Details'].str.contains('brush').sum()
alarm_count_all = df['Risk Details'].str.contains('alarm').sum()
updated_count_all = df['Risk Details'].str.contains('update').sum()
roof_count_all = df['Risk Details'].str.contains('roof').sum()
loss_count_all = df['Risk Details'].str.contains('loss').sum()
fire_count_all = df['Risk Details'].str.contains('fire').sum()
response_time_count_all = df['Risk Details'].str.contains('response time').sum()
online_photo_count_all = df['Risk Details'].str.contains('photo').sum()

risk_counts = pd.DataFrame({"Risk_Types":["Brush Count", "Alarm Count", "Home Update Count", "Roof Count", "Loss Count", "Fire Count", "Response Time Count", "Online Photo Count"],
                            "Risk_Counts":[brush_count_all, alarm_count_all, updated_count_all, roof_count_all, loss_count_all, fire_count_all, response_time_count_all, online_photo_count_all]})

print(brush_count_all+alarm_count_all+updated_count_all+loss_count_all+fire_count_all+response_time_count_all+online_photo_count_all)

risk_counts = risk_counts.sort_values(['Risk_Counts'], ascending=False).reset_index(drop=True)

Risk_Types = sns.barplot(x='Risk_Counts', y='Risk_Types', data=risk_counts)
Risk_Types.set(xlabel = 'Frequency', ylabel = 'Risk Words', title = 'Risk Words Appearing by Frequency (Risk Details Field)')
plt.show()

## Find risk factor occurences in RISK DETAILS (DO NOT decline)

brush_count = approvalDF['Risk Details'].str.contains('brush').sum()
alarm_count = approvalDF['Risk Details'].str.contains('alarm').sum()
updated_count = approvalDF['Risk Details'].str.contains('update').sum()
roof_count = approvalDF['Risk Details'].str.contains('roof').sum()
loss_count = approvalDF['Risk Details'].str.contains('loss').sum()
fire_count = approvalDF['Risk Details'].str.contains('fire').sum()
response_time_count = approvalDF['Risk Details'].str.contains('response time').sum()
online_photo_count = approvalDF['Risk Details'].str.contains('photo').sum()

risk_counts_approval = pd.DataFrame({"Risk_Types":["Brush Count", "Alarm Count", "Home Update Count", "Roof Count", "Loss Count", "Fire Count", "Response Time Count", "Online Photo Count"],
                            "Approval_Risk_Counts":[brush_count, alarm_count, updated_count, roof_count, loss_count, fire_count, response_time_count, online_photo_count]})

risk_counts_approval = risk_counts_approval.sort_values(['Approval_Risk_Counts'], ascending=False).reset_index(drop=True)

App_Risk_Types = sns.barplot(x='Approval_Risk_Counts', y='Risk_Types', data=risk_counts_approval, color='firebrick')
App_Risk_Types.set(xlabel = 'Frequency', ylabel = 'Risk Words', title = 'Risk Words Appearing in Approvals by Frequency (Risk Details Field)')
plt.show()

combined = pd.merge(risk_counts, risk_counts_approval, how='left', on='Risk_Types')
combined = pd.merge(combined, risk_counts_decline, how='left', on='Risk_Types')
combined['approval %'] = combined['Approval_Risk_Counts'] / combined['Risk_Counts'] * 100
combined['decline %'] = combined['Declined_Risk_Counts'] / combined['Risk_Counts'] * 100

## Find risk factor occurences in RISK DETAILS (DO decline)

brush_count_decline = declineDF['Risk Details'].str.contains('brush').sum()
alarm_count_decline = declineDF['Risk Details'].str.contains('alarm').sum()
updated_count_decline = declineDF['Risk Details'].str.contains('update').sum()
roof_count_decline = declineDF['Risk Details'].str.contains('roof').sum()
loss_count_decline = declineDF['Risk Details'].str.contains('loss').sum()
fire_count_decline = declineDF['Risk Details'].str.contains('fire').sum()
response_time_count_decline = declineDF['Risk Details'].str.contains('response time').sum()
online_photo_count_decline = declineDF['Risk Details'].str.contains('photo').sum()

risk_counts_decline = pd.DataFrame({"Risk_Types":["Brush Count", "Alarm Count", "Home Update Count", "Roof Count", "Loss Count", "Fire Count", "Response Time Count", "Online Photo Count"],
                            "Declined_Risk_Counts":[brush_count_decline, alarm_count_decline, updated_count_decline, roof_count_decline, loss_count_decline, fire_count_decline,
                                                    response_time_count_decline, online_photo_count_decline]})

risk_counts_decline = risk_counts_decline.sort_values(['Declined_Risk_Counts'], ascending=False).reset_index(drop=True)

Dec_Risk_Types = sns.barplot(x='Declined_Risk_Counts', y='Risk_Types', data=risk_counts_decline)
Dec_Risk_Types.set(xlabel = 'Frequency', ylabel = 'Risk Words', title = 'Risk Words Appearing in Declines by Frequency (Risk Details Field)')
plt.show()

## Find differences around risk words in DO NOT decline and DO decline

app_loss = approvalDF['Risk Details'].str.contains('no loss').sum()
dec_loss = declineDF['Risk Details'].str.contains('no loss').sum()

loss_hist = pd.DataFrame({"No_Loss_Policies":["Approvals No Loss Count", "Declines No Loss Count"],"No_Loss_Counts":[app_loss, dec_loss]})
App_Dec_Loss = sns.barplot(x='No_Loss_Policies',y='No_Loss_Counts', data=loss_hist)
App_Dec_Loss.set(xlabel='Decision Type', ylabel='No Loss Count', title='Policies with No Losses')
plt.show()

print(approvalDF['Risk Details'].str.contains('no update').sum() + approvalDF['Risk Details'].str.contains('not updated').sum())
print(declineDF['Risk Details'].str.contains('no update').sum() + declineDF['Risk Details'].str.contains('not updated').sum())

print(approvalDF['Risk Details'].str.contains('no online photo').sum())
print(declineDF['Risk Details'].str.contains('no online photo').sum())


## Analyze Home Update Risk (DO NOT decline)

home_update_approval = approvalDF[approvalDF['Risk Details'].str.contains('update')]
p = re.compile('[1][0-9]{3}')
def get_old_years(x):
    if len(p.findall(x)) == 1:
        return p.findall(x)
home_years = home_update_approval['Risk Details'].apply(get_old_years)
old_home_years = []
for val in home_years:
    if val != None :
        old_home_years.append(val)
old_home_years = pd.Series((v[0] for v in old_home_years))
old_home_years = old_home_years.apply(lambda x: int(x))


## Analyze Home Update Risk (DO Decline)

home_update_decline = declineDF[declineDF['Risk Details'].str.contains('update')]
home_years_decline = home_update_decline['Risk Details'].apply(get_old_years)
old_home_years_decline = []
for val in home_years_decline:
    if val != None :
        old_home_years_decline.append(val)
old_home_years_decline = pd.Series((v[0] for v in old_home_years_decline))
old_home_years_decline = old_home_years_decline.apply(lambda x: int(x))

## Plot both distributions

app_home = sns.distplot(old_home_years, label='Approvals Old Homes', color='firebrick')
dec_home = sns.distplot(old_home_years_decline, label='Declines Old Homes', color='black')

app_home.set(xlabel='Home Years', ylabel='Frequency', title='Frequency Distribution of Old Home Years')

plt.legend()
plt.show()

## Bigrams

approvalDF['Risk Details'] = approvalDF['Risk Details'].str.replace('\d+', '')
declineDF['Risk Details'] = declineDF['Risk Details'].str.replace('\d+', '')
df['Reason(s) for Decision'] = df['Reason(s) for Decision'].str.replace('\d+', '')
df['Risk Details'] = df['Risk Details'].str.replace('\d+', '')

## All Reasons for Decision

word_vectorizer5 = CountVectorizer(ngram_range=(2, 3), analyzer='word')
sparse_matrix5 = word_vectorizer5.fit_transform(df['Reason(s) for Decision'])
frequencies5 = sum(sparse_matrix5).toarray()[0]
tok5 = pd.DataFrame(frequencies5, index=word_vectorizer5.get_feature_names(), columns=['frequency'])
tok5 = tok5.sort_values(by=['frequency'], ascending=False)

## All Risk Details

word_vectorizer6 = CountVectorizer(ngram_range=(2, 3), analyzer='word')
sparse_matrix6 = word_vectorizer6.fit_transform(df['Risk Details'])
frequencies6 = sum(sparse_matrix6).toarray()[0]
tok6 = pd.DataFrame(frequencies6, index=word_vectorizer6.get_feature_names(), columns=['frequency'])
tok6 = tok6.sort_values(by=['frequency'], ascending=False)

## Approval Risk Details Bigrams
word_vectorizer3 = CountVectorizer(ngram_range=(2,3), analyzer='word')
sparse_matrix3 = word_vectorizer3.fit_transform(approvalDF['Risk Details'])
frequencies3 = sum(sparse_matrix3).toarray()[0]
tok3 = pd.DataFrame(frequencies3, index=word_vectorizer3.get_feature_names(), columns=['frequency'])
tok3 = tok3.sort_values(by=['frequency'], ascending=False)

## Decline Risk Details Bigrams
word_vectorizer = CountVectorizer(ngram_range=(2,3), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(declineDF['Risk Details'])
frequencies = sum(sparse_matrix).toarray()[0]
tok = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])
tok = tok.sort_values(by=['frequency'], ascending=False)

## Approval Reason for Decision Bigrams
word_vectorizer2 = CountVectorizer(ngram_range=(2, 3), analyzer='word')
sparse_matrix2 = word_vectorizer2.fit_transform(approvalDF['Reason(s) for Decision'])
frequencies2 = sum(sparse_matrix2).toarray()[0]
tok2 = pd.DataFrame(frequencies2, index=word_vectorizer2.get_feature_names(), columns=['frequency'])
tok2 = tok2.sort_values(by=['frequency'], ascending=False)

## Decline Reason for Decision Bigrams
word_vectorizer4 = CountVectorizer(ngram_range=(2, 3), analyzer='word')
sparse_matrix4 = word_vectorizer4.fit_transform(declineDF['Reason(s) for Decision'])
frequencies4 = sum(sparse_matrix4).toarray()[0]
tok4 = pd.DataFrame(frequencies4, index=word_vectorizer4.get_feature_names(), columns=['frequency'])
tok4 = tok4.sort_values(by=['frequency'], ascending=False)

## Decline Bigram Chart (Risk Details)
tok5 = tok.iloc[[1,2,7,11,19]]
dec_bi_det = sns.barplot(x=tok5.index.values, y='frequency', data=tok5, color='firebrick')
dec_bi_det.set(xlabel='Top Bigrams', ylabel='Frequency', title='Top Bigrams in Declines (Risk Details)')
plt.show()

## Approval Bigram Chart (Risk Details)
tok7 = tok3.iloc[[1,2,4,10,11]]
app_bi_det = sns.barplot(x=tok7.index.values, y='frequency', data=tok7, color='firebrick')
app_bi_det.set(xlabel='Top Bigrams', ylabel='Frequency', title='Top Bigrams in Approvals (Risk Details)')
plt.show()


## Analyze Response Time Risk (DO NOT Decline)

response_time_approval = approvalDF[approvalDF['Risk Details'].str.contains('response time')]
rt1 = re.compile('(\d{2}).{3,15}response time')
rt2 = re.compile('response time.{3,15}(\d{2})')
def get_response_times1(x):
    if len(rt1.findall(x)) == 1:
        return rt1.findall(x)
def get_response_times2(x):
    if len(rt2.findall(x)) == 1:
        return rt2.findall(x)
response_times1 = response_time_approval['Risk Details'].apply(get_response_times1)
response_times2 = response_time_approval['Risk Details'].apply(get_response_times2)
app_resp = []
for val in response_times1:
    if val != None :
        app_resp.append(val)
for val in response_times2:
    if val != None :
        app_resp.append(val)
approval_response_times = pd.Series((v[0] for v in app_resp))
approval_response_times = approval_response_times.sort_values(ascending=False)
approval_response_times = approval_response_times.apply(lambda x: int(x))
approval_response_times = approval_response_times[approval_response_times < 30]

## Analyze Response Time Risk (DO Decline)

response_time_decline = declineDF[declineDF['Risk Details'].str.contains('response time')]
response_times3 = response_time_decline['Risk Details'].apply(get_response_times1)
response_times4 = response_time_decline['Risk Details'].apply(get_response_times2)
dec_resp = []
for val in response_times3:
    if val != None :
        dec_resp.append(val)
for val in response_times4:
    if val != None :
        dec_resp.append(val)
declined_response_times = pd.Series((v[0] for v in dec_resp))
declined_response_times = declined_response_times.sort_values(ascending=False)
declined_response_times = declined_response_times.apply(lambda x: int(x))
declined_response_times = declined_response_times[declined_response_times > 10]

## Plot both distributions

app_resp = sns.distplot(approval_response_times, label='Approvals Response Times', color='firebrick')
dec_resp = sns.distplot(declined_response_times, label='Declines Response Times', color='black')

app_resp.set(xlabel='Response Times', ylabel='Frequency', title='Frequency Distribution of Response Times')

plt.legend()
plt.show()

## Need to add features to the dataset (each risk type)

## Response Time (int)

print(len(df))

## In Brush (0/1)

## Losses (int)

## Home Year (int)

## Systems Update Year (int)

## Online Photo (0/1)

