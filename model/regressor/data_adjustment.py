import pandas as pd
import numpy as np
import numbers


def adjust_data(client_a_df):
    for item in client_a_df.columns:
        if isinstance(client_a_df[item][1], numbers.Number):
            client_a_df[item] = pd.to_numeric(client_a_df[item])

    client_a_df.role_level = pd.to_numeric(client_a_df.role_level, errors='coerce')
    client_a_df.total_bonus_amount = pd.to_numeric(client_a_df.total_bonus_amount, errors='coerce')
    client_a_df.hours_per_week = pd.to_numeric(client_a_df.hours_per_week, errors='coerce')
    client_a_df.career_level = pd.to_numeric(client_a_df.career_level, errors='coerce')

    years = []
    for service in client_a_df.service_length:
        years.append(int(service.split()[0]))

    client_a_df['service_length_years'] = years

    period = []
    for service in client_a_df.service_length:
        if int(service.split()[0]) == 0:
            period.append('<1')
        elif 1 <= int(service.split()[0]) <= 5:
            period.append('1-5')
        elif 6 <= int(service.split()[0]) <= 10:
            period.append('6-10')
        elif 11 <= int(service.split()[0]) <= 15:
            period.append('11-15')
        elif 16 <= int(service.split()[0]) <= 20:
            period.append('16-20')
        elif 21 <= int(service.split()[0]) <= 25:
            period.append('21-25')
        elif 26 <= int(service.split()[0]) <= 30:
            period.append('26-30')
        else:
            period.append('30<')

    client_a_df['service_length_category'] = period

    tenure = []
    for ten in client_a_df.company_tenure:
        if ten == 0:
            tenure.append('<1')
        elif 1 <= ten <= 5:
            tenure.append('1-5')
        elif 6 <= ten <= 10:
            tenure.append('6-10')
        elif 11 <= ten <= 15:
            tenure.append('11-15')
        elif 16 <= ten <= 20:
            tenure.append('16-20')
        elif 21 <= ten <= 25:
            tenure.append('21-25')
        else:
            tenure.append('25<')

    client_a_df['company_tenure_category'] = tenure

    month_salaries = []
    for sal in client_a_df.salary_per_month_35ft:
        if sal < 1500:
            month_salaries.append('<1500')
        elif 1500 <= sal < 2000:
            month_salaries.append('1500-2000')
        elif 2000 <= sal < 3000:
            month_salaries.append('2000-3000')
        elif 3000 <= sal < 4000:
            month_salaries.append('3000-4000')
        elif 4000 <= sal < 5000:
            month_salaries.append('4000-5000')
        elif 5000 <= sal < 6000:
            month_salaries.append('5000-6000')
        elif 6000 <= sal < 7000:
            month_salaries.append('6000-7000')
        elif 7000 <= sal < 8000:
            month_salaries.append('7000-8000')
        elif 8000 <= sal < 9000:
            month_salaries.append('8000-9000')
        elif 10000 <= sal < 15000:
            month_salaries.append('10000-15000')
        else:
            month_salaries.append('15000<')

    client_a_df['35ft_equivalent_monthly_category'] = month_salaries

    if_bonus = []

    for bonus in client_a_df.total_bonus_amount:
        if bonus == 'No bonus':
            if_bonus.append(False)
        else:
            if_bonus.append(True)

    client_a_df['if_bonus'] = if_bonus

    client_a_df = client_a_df.round(decimals=2)

    return client_a_df
