"""
Hide Sensitive Data and replace client and team names
"""

import pandas as pd
import sys
import psycopg2
import numpy as np
from len_service_change import service_float
from data_from_postgresql import postgresql_to_dataframe

try:
    # Connect Database
    conn = psycopg2.connect(dbname='', user='', host='', port='',
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

    df = postgresql_to_dataframe(conn, "select * from client_a", col_names)

except psycopg2.DatabaseError:
    sys.exit('Failed to connect to database')
finally:
    if conn is not None:
        conn.close()


df = df[['division', 'groups', 'area', 'team', 'job_title', 'category', 'company_tenure', 'hourly_equivalent',
        'notice_rule', 'gender', 'work_style', 'qualified_solicitor', 'appointment_type', 'career_level',
        'role_level', 'ft_pay_amount', 'post_name', 'benefit_level', 'total_bonus_amount', 'city', 'service_length']]

# Replace the values with generic incremental values
for item in ['division', 'groups', 'area', 'team', 'job_title', 'category', 'post_name', 'benefit_level']:
    uniq_vals = pd.unique(df[item])
    anonym_vals = ['{}_{}'.format(item, i) for i in range(1, len(uniq_vals)+1)]
    replace_dict = {uniq_vals[j]: anonym_vals[j] for j in range(len(uniq_vals))}
    df = df.replace({item: replace_dict})

df = service_float(df)

df.to_csv('client_x_anonym.csv')
