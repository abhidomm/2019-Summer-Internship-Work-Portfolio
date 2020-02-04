## THIS PYTHON FILE IS AN ANLYSIS OF THE IMPACT THE UNDERWRITER WORKBOOK HAS ON BUSINESS VALUE
## METRICS LIKE POLICY COUNT, TIME SAVINGS, AND GWP EXPLORED

## IMPORT STATEMENTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## READ IN ANALYSIS DATA

df = pd.read_csv('Workbook_Analysis_Data.csv')

## POLICY COUNTS BY PRODUCT LINE

no_binding_df = df[df['product_line'] != 'Binding P&C']
pol_count = sns.countplot(x='product_line', data=no_binding_df, order=no_binding_df['product_line'].value_counts().index, color='firebrick')
pol_count.set(xlabel='Product Lines', ylabel='Policy Count', title='Policy Counts by Product Line')
plt.show()

print(len(df[df['product_line'] == 'Comm Primary Cas Wholesale']) + len(df[df['product_line'] == 'Healthcare Risk Solutions']) + len(df[df['product_line'] == 'Ocean Marine']))

## AVERAGE GWP BY PRODUCT LINE

result = no_binding_df.groupby(['product_line'])['gwp'].aggregate(np.mean).reset_index().sort_values('gwp', ascending=False)
avg_gwp = sns.barplot(x='product_line', y='gwp', data=no_binding_df, order=result['product_line'], color='firebrick')
avg_gwp.set(xlabel='Product Lines', ylabel='Average GWP', title='Average GWP by Product Line')
plt.show()

## AVERAGE LOSS EXPERIENCE BY PRODUCT LINE

avg_loss_exp = sns.barplot(x='product_line', y='loss_exp', data=df)
avg_loss_exp.set(xlabel='Product Lines', ylabel='Average Loss Experience', title='Average Loss Experience by Product Line')
plt.show()

## AVERAGE ACCOUNT YEARS BY PRODUCT LINE

result = df.groupby(['product_line'])['account_years'].aggregate(np.mean).reset_index().sort_values('account_years', ascending=False)
avg_acc_y = sns.barplot(x='product_line', y='account_years', data=df, order=result['product_line'])
avg_acc_y.set(xlabel='Product Lines', ylabel='Average Account Years', title='Average Account Years by Product Line')
plt.show()

## AVERAGE NUMBER OF CLAIMS BY PRODUCT LINE

result = df.groupby(['product_line'])['total_claim'].aggregate(np.mean).reset_index().sort_values('total_claim', ascending=False)
avg_claims = sns.barplot(x='product_line', y='total_claim', data=df, order=result['product_line'])
avg_claims.set(xlabel='Product Lines', ylabel='Average Number of Claims', title='Average Number of Claims by Product Line')
plt.show()

## AVERAGE NUMBER OF OPEN CLAIMS BY PRODUCT LINE

open_claims = sns.barplot(x='product_line', y='open_claim', data=df)
open_claims.set(xlabel='Product Lines', ylabel='Average Number of Open Claims', title='Average Number of Open Claims by Product Line')
plt.show()

## CLAIMS PORTIONS BY PRODUCT LINE

claims_port = sns.countplot(x='product_line', hue='has_claims', data=df)
claims_port.set(xlabel='Product Lines', ylabel='Yes/No Claim Count', title='Claims Portions by Product Line')
plt.show()

## XRO FLAG PORTIONS BY PRODUCT LINE

no_binding_df = df[df['product_line'] != 'Binding P&C']
result = no_binding_df.groupby(['product_line'])['xro_flag'].aggregate(np.mean).reset_index().sort_values('xro_flag', ascending=False)
xro_port = sns.countplot(x='product_line', hue='xro_flag', data=no_binding_df, order=result['product_line'])
xro_port.set(xlabel='Product Lines', ylabel='Yes/No XRO Flag Count', title=' XRO Flag Portions by Product Line')
plt.show()

print(no_binding_df['gwp'].sum())

## BROKERAGE CASUALTY RATE GUIDANCE

bc_rate = df[df['BC_action_taken'] == 1]

## DESIRE TO RETAIN POLICY

bc_retain_desire = sns.countplot(x='BC_desire_retain', data=bc_rate, order=bc_rate['BC_desire_retain'].value_counts().index)
bc_retain_desire.set(xlabel='Retain Desire', ylabel='Count', title='Desire to Retain Brokerage Casualty Policies')
plt.show()

## DECISION BASIS FOR ACTION TAKEN

bc_decision_basis = sns.countplot(x='BC_decision', data=bc_rate)
bc_decision_basis.set(xlabel='Decision Basis', ylabel='Count', title='Decision Basis for Rate Guidance Action')
plt.show()

## GWP BY RETAIN DESIRE

bc_GWP = sns.barplot(x='BC_desire_retain', y='gwp', data=bc_rate)
bc_GWP.set(xlabel='Retain Desire', ylabel='Average GWP', title='Average GWP by Brokerage Casualty by Retain Desires')
plt.show()

## OFFERED RATE FLOOR BY RATE DESIRE

bc_rate_floor = sns.barplot(x='BC_desire_retain', y='BC_offered_rate_floor', data=bc_rate)
bc_rate_floor.set(xlabel='Retain Desire', ylabel='Average Offered Rate Floor', title='Average Offered Rate Floor by Retain Desires')
plt.show()

## PREMIUM UPDATE BY RETAIN DESIRE

bc_premium_update = sns.barplot(x='BC_desire_retain', y='BC_premium_update', data=bc_rate)
bc_premium_update.set(xlabel='Retain Desire', ylabel='Average Premium Update', title='Average Premium Update by Retain Desire')
plt.show()

## RISK SCORE UPDATE BY RETAIN DESIRE

bc_risk_update = sns.barplot(x='BC_desire_retain', y='BC_risk_score_update', data=bc_rate)
bc_risk_update.set(xlabel='Retain Desire', ylabel='Average Risk Score Update', title='Average Risk Score Update by Retain Desire')
plt.show()

## LOSS UPDATE BY RETAIN DESIRE

bc_loss_update = sns.barplot(x='BC_desire_retain', y='BC_loss_update', data=bc_rate)
bc_loss_update.set(xlabel='Retain Desire', ylabel='Average Loss Update', title='Average Loss Update by Retain Desire')
plt.show()

## TIME SAVINGS DATA ANALYSIS INCORPORATED
with_claims_df = df[df['has_claims'] == 1]

## PRIMARY CASUALTY

prim_cas_before_time = 20
prim_cas_after_time = 5

prim_cas_pol_count = len(with_claims_df[with_claims_df['product_line'] == 'Comm Primary Cas Wholesale'])

prim_cas_before_pol_time = prim_cas_before_time * prim_cas_pol_count / 60
prim_cas_after_pol_time = prim_cas_after_time * prim_cas_pol_count / 60

prim_cas_time_savings_pol = pd.DataFrame({"Product Line": ["Comm Primary Cas Wholesale", "Comm Primary Cas Wholesale"],"Before/After":["Time Spent Before", "Time Spent After"], "Time Spent":[prim_cas_before_pol_time, prim_cas_after_pol_time]})

prim_cas_pol_time_saved = sns.barplot(x='Before/After', y='Time Spent', data=prim_cas_time_savings_pol)
prim_cas_pol_time_saved.set(xlabel='Before and After the Workbook', ylabel='Time Spent (hours)', title='Total Time Spent on Primary Casaulty Policies with Claims')
plt.show()

prim_cas_avg_gwp = (with_claims_df[with_claims_df['product_line'] == 'Comm Primary Cas Wholesale'])['gwp'].mean()

prim_cas_before_gwp_per_min = prim_cas_avg_gwp / prim_cas_before_time
prim_cas_after_gwp_per_min = prim_cas_avg_gwp / prim_cas_after_time

prim_cas_time_savings_gwp = pd.DataFrame({"Product Line": ["Comm Primary Cas Wholesale", "Comm Primary Cas Wholesale"],"Before/After":["GWP/MIN Acquired Before", "GWP/MIN Acquired After"], "GWP Dollar Acquired":[prim_cas_before_gwp_per_min, prim_cas_after_gwp_per_min]})

prim_cas_GWP_acquired = sns.barplot(x='Before/After', y='GWP Dollar Acquired', data=prim_cas_time_savings_gwp)
prim_cas_GWP_acquired.set(xlabel='Before and After the Workbook', ylabel='GWP Dollars Acquired', title='Average GWP Dollars Acquired per Minute on Primary Casualty Policies')
plt.show()

## HEALTHCARE RISK SOLUTIONS

health_before_time = 60
health_after_time = 5

health_pol_count = len(with_claims_df[with_claims_df['product_line'] == 'Healthcare Risk Solutions'])

health_before_pol_time = health_before_time * health_pol_count / 60
health_after_pol_time = health_after_time * health_pol_count / 60

health_time_savings_pol = pd.DataFrame({"Product Line":["Healthcare Risk Solutions", "Healthcare Risk Solutions"],"Before/After":["Time Spent Before", "Time Spent After"], "Time Spent":[health_before_pol_time, health_after_pol_time]})

health_pol_time_saved = sns.barplot(x='Before/After', y='Time Spent', data=health_time_savings_pol)
health_pol_time_saved.set(xlabel='Before and After the Workbook', ylabel='Time Spent (hours)', title='Total Time Spent on Healthcare Policies with Claims')
plt.show()

health_avg_gwp = (with_claims_df[with_claims_df['product_line'] == 'Healthcare Risk Solutions'])['gwp'].mean()

health_before_gwp_per_min = health_avg_gwp / health_before_time
health_after_gwp_per_min = health_avg_gwp / health_after_time

health_time_savings_gwp = pd.DataFrame({"Product Line":["Healthcare Risk Solutions", "Healthcare Risk Solutions"],"Before/After":["GWP/MIN Acquired Before", "GWP/MIN Acquired After"], "GWP Dollar Acquired":[health_before_gwp_per_min, health_after_gwp_per_min]})

health_GWP_acquired = sns.barplot(x='Before/After', y='GWP Dollar Acquired', data=health_time_savings_gwp)
health_GWP_acquired.set(xlabel='Before and After the Workbook', ylabel='GWP Dollars Acquired', title='Average GWP Dollars Acquired per Minute on Healthcare Policies with Claims')
plt.show()

## OCEAN MARINE

ocean_before_time = 20
ocean_after_time = 3

ocean_pol_count = len(with_claims_df[with_claims_df['product_line'] == 'Ocean Marine'])

ocean_before_pol_time = ocean_before_time * ocean_pol_count / 60
ocean_after_pol_time = ocean_after_time * ocean_pol_count / 60

ocean_time_savings_pol = pd.DataFrame({"Product Line":["Ocean Marine", "Ocean Marine"], "Before/After":["Time Spent Before", "Time Spent After"], "Time Spent":[ocean_before_pol_time, ocean_after_pol_time]})

ocean_pol_time_saved = sns.barplot(x='Before/After', y='Time Spent', data=ocean_time_savings_pol)
ocean_pol_time_saved.set(xlabel='Before and After the Workbook', ylabel='Time Spent (hours)', title='Total Time Spent on Ocean Marine Policies with Claims')
plt.show()

ocean_avg_gwp = (with_claims_df[with_claims_df['product_line'] == 'Ocean Marine'])['gwp'].mean()

ocean_before_gwp_per_min = ocean_avg_gwp / ocean_before_time
ocean_after_gwp_per_min = ocean_avg_gwp / ocean_after_time

ocean_time_savings_gwp = pd.DataFrame({"Product Line":["Ocean Marine", "Ocean Marine"],"Before/After":["GWP/MIN Acquired Before", "GWP/MIN Acquired After"], "GWP Dollar Acquired":[ocean_before_gwp_per_min, ocean_after_gwp_per_min]})

ocean_GWP_acquired = sns.barplot(x='Before/After', y='GWP Dollar Acquired', data=ocean_time_savings_gwp)
ocean_GWP_acquired.set(xlabel='Before and After the Workbook', ylabel='GWP Dollars Acquired', title='Average GWP Dollars Acquired per Minute on Ocean Marine Policies with Claims')
plt.show()

## COMBINED TOTAL TIME AND GWP/MIN METRIC

all_PL_time_savings_pol = prim_cas_time_savings_pol.append(health_time_savings_pol).append(ocean_time_savings_pol)
time_savings_pol = sns.barplot(x='Product Line', y='Time Spent', data=all_PL_time_savings_pol, hue='Before/After', palette='Reds')
time_savings_pol.set(xlabel='Product Line', ylabel='Time Spent (hours)', title='Total Time Spent on Policies with Claims')

all_PL_time_savings_gwp = prim_cas_time_savings_gwp.append(health_time_savings_gwp).append(ocean_time_savings_gwp)
time_savings_gwp = sns.barplot(x='Product Line', y='GWP Dollar Acquired', data=all_PL_time_savings_gwp, hue='Before/After', palette='Reds')
time_savings_gwp.set(xlabel='Product Line', ylabel='GWP Dollars Acquired', title='Average GWP Dollars Acquired per Minute on Policies with Claims')

