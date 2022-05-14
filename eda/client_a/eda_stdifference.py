# %%
# !/usr/bin/env python3
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import sys
import psycopg2
import pandas as pd
import numpy as np
import numbers

# visualisations
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import style
style.use('fivethirtyeight')
import seaborn as sns
import shap

# modelling helpers
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.pipeline import Pipeline

from data_from_postgresql import postgresql_to_dataframe
from data_adjustment import adjust_data
from predict_salary_helper import help_norm_salary



# %%
def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()

# %%
try:
    conn = psycopg2.connect(dbname='gapsquare', user='', host='127.0.0.1', port='5432',
                            password='')
    print('Connection successful')

    col_names = ['employee', 'division', 'groups', 'area', 'team', 'location', 'job_title', 'status', 'category',
                 'join_date', 'company_tenure',
                 'service_length', 'pay_amount', 'hourly_equivalent', 'notice_rule', 'ethnic_origin', 'nationality',
                 'gender', 'disabled', 'work_style',
                 'days_per_week', 'contract', 'appointment_type', 'marital_status', 'qualified_solicitor',
                 'benefit_level', 'full_time_equivalent',
                 'currency', 'ft_hours', 'ft_pay_amount', 'wp_description', 'career_effective_date',
                 'career_change_reason', 'pay_from_date',
                 'pay_change_reason', 'non_executive_director', 'payroll_name', 'tax', 'net_pay', 'basic_pay_nl_uk',
                 'payroll', 'post_name',
                 'arc_pension', 'auto_daily_rate', 'auto_hourly_rate', 'ees_nic', 'total_gross', 'city',
                 'salary_per_week_35ft', 'salary_per_month_35ft',
                 'salary_annual_35ft', 'salary_plus_bonus_35ft', 'role_level', 'career_level', 'hours_per_week',
                 'total_bonus_amount']

    client_a_df = postgresql_to_dataframe(conn, "select * from client_a", col_names)

except psycopg2.DatabaseError:
    sys.exit('Failed to connect to database')
finally:
    if conn is not None:
        conn.close()


# %%
client_a_df = client_a_df.drop('currency', axis=1)
client_a_df = client_a_df.drop('non_executive_director', axis=1)

# %%
print('Females: ', client_a_df[client_a_df.gender == 'Female'].count()[1])

# %%
print('Males: ', client_a_df[client_a_df.gender == 'Male'].count()[1])

# %%
client_a_df = adjust_data(client_a_df)


#%%
for item in client_a_df.columns:
    if item not in ('join_date', 'company_tenure', 'service_length', 'pay_amount', 'hourly_equivalent', 'ft_pay_amount',
                    'wp_description', 'career_effective_date', 'pay_from_date', 'tax', 'net_pay', 'basic_pay_nl_uk',
                    'arc_pension', 'auto_daily_rate', 'auto_hourly_rate', 'ees_nic', 'total_gross', 'salary_annual_35ft',
                    'salary_per_month_35ft', 'salary_per_week_35ft', 'salary_plus_bonus_35ft', 'total_bonus_amount'):
        plot_distribution(client_a_df, 'salary_per_month_35ft', item)
        plt.title('Monthly salary distribution vs %s' % item)
        plt.savefig('dist_month_salary_%s.png' % item, bbox_inches='tight')
        plt.show()



# %%
for item in client_a_df.columns:
    if item in ('division', 'groups', 'area', 'team', 'location', 'status', 'category', 'notice_rule', 'ethnic_origin',
                'nationality', 'gender', 'disabled', 'work_style', 'contract', 'appointment_type', 'marital_status',
                'qualified_solicitor', 'benefit_level', 'full_time_equivalent', 'ft_hours', 'career_change_reason',
                'pay_change_reason', 'payroll_name', 'payroll', 'post_name', 'city', 'role_level', 'career_level',
                'hours_per_week', 'service_length_category', 'company_tenure_category',
                '35ft_equivalent_monthly_category', 'if_bonus'):
        plt.figure(figsize=(15, 15))
        sns_plot = sns.catplot(x=item, hue='gender', kind="count", data=client_a_df)
        plt.title('Histogram of %s' % item)
        plt.xticks(rotation=90)
        sns_plot.savefig('%s_histogram.png' % item, bbox_inches='tight')
        plt.show()

#%%
for item in ('service_length_years', 'career_level', 'total_gross', 'ees_nic', 'company_tenure'):
    plt.scatter(client_a_df[item], client_a_df['salary_per_month_35ft'])
    plt.title('Monthly salary (35FTE) vs. %s' % item)
    plt.xlabel(item)
    plt.ylabel('Monthly salary 35FTE')
    plt.savefig('month_salary_vs_%s.png' % item, bbox_inches='tight')
    plt.show()

#%%
for item in ('city', 'division', 'status', 'category', 'work_style', 'contract', 'appointment_type', 'marital_status',
             'qualified_solicitor', 'benefit_level', 'career_change_reason', 'pay_change_reason', 'payroll_name',
             'payroll', 'service_length_category', 'company_tenure_category', 'if_bonus'):
    plt.figure(figsize=(15, 10))
    sns.violinplot(x=item, y='salary_per_week_35ft', hue='gender', data=client_a_df, split=True)
    plt.ylabel("Monthly Income FTE", fontsize=12)
    plt.xticks(rotation=-90)
    plt.xlabel(item, fontsize=12)
    plt.title("Illustration of the Gender Wage for Different %s" % item, fontsize=15)
    plt.savefig('violin_%s.png' % item, bbox_inches='tight')
    plt.show()



#%%
for item in client_a_df.columns:
    # if item in ('city', 'division', 'groups', 'location', 'marital_status', 'pay_label', 'payroll', 'payroll_name',
    #             'qualified_solicitor', 'service_length', 'work_style'):
    if item == 'service_length_category':
        plt.figure(figsize=(15, 15))
        sns.boxplot(x=item, y='salary_per_month_35ft', hue='gender', palette='husl', data=client_a_df, showfliers=False,
                    order=['<1', '1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '30<'])
        plt.xticks(rotation=-70)
        plt.ylabel("Monthly Income FTE", fontsize=12)
        plt.xlabel(item, fontsize=12)
        plt.title("Illustration of the Wage for Different %s per gender" % item, fontsize=15)
        plt.savefig('boxplot_%s_gender.png' % item, bbox_inches='tight')
        plt.show()


#%%
client_a_df = help_norm_salary(client_a_df)


#%%
m_job = max(client_a_df.norm_salary_job_title)
for value in client_a_df.norm_salary_job_title:
    if value == m_job:
        print(value)
        print(client_a_df.index[client_a_df['norm_salary_job_title'] == value].tolist())


#%%
for item in client_a_df.columns:
    # if item in ('division', 'groups', 'area', 'team', 'location', 'status', 'category', 'notice_rule', 'ethnic_origin',
    #             'nationality', 'gender', 'disabled', 'work_style', 'contract', 'appointment_type', 'marital_status',
    #             'qualified_solicitor', 'benefit_level', 'full_time_equivalent', 'ft_hours', 'career_change_reason',
    #             'pay_change_reason', 'payroll_name', 'payroll', 'post_name', 'city', 'role_level', 'career_level',
    #             'hours_per_week', 'service_length_category', 'company_tenure_category',
    #             '35ft_equivalent_monthly_category', 'if_bonus'):
    if item == '35ft_equivalent_monthly_category':
        plt.figure(figsize=(15, 15))
        sns_plot = sns.catplot(x=item, hue='gender', kind="count", data=client_a_df, order=['<1500', '1500-2000',
                                                                                            '2000-3000', '3000-4000',
                                                                                            '4000-5000', '5000-6000',
                                                                                            '6000-7000', '7000-8000',
                                                                                            '8000-9000', '10000-15000',
                                                                                            '15000<'])
        plt.title('Histogram of %s' % item)
        plt.xticks(rotation=90)
        sns_plot.savefig('%s_histogram.png' % item, bbox_inches='tight')
        plt.show()
