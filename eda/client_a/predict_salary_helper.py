import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def help_norm_salary(client_a_df):
    for item in ('division', 'area', 'team', 'location', 'job_title', 'category', 'company_tenure', 'service_length',
                 'notice_rule', 'contract', 'appointment_type', 'benefit_level', 'payroll_name', 'payroll', 'post_name',
                 'city', 'role_level', 'career_level'):
        average_val = client_a_df.groupby(item, as_index=False)['salary_per_month_35ft'].mean()
        # normalised = NormalizeData(average_val.salary_per_month_35ft)
        # average_val['normalised_salary'] = normalised
        # average_val = average_val.drop(columns='salary_per_month_35ft')
        average_val_dict = pd.Series(average_val.salary_per_month_35ft.values, index=average_val[item]).to_dict()

        client_a_df.insert(loc=client_a_df.columns.get_loc(item) + 1, column='norm_salary_%s' % item,
                           value=[np.nan for i in range(client_a_df.shape[0])])

        client_a_df["norm_salary_%s" % item] = client_a_df[item].apply(lambda x: average_val_dict.get(x))

    norm_values = []
    for col in client_a_df.columns:
        if col[:4] == 'norm':
            norm_values.append(col)

    average_salary = client_a_df[norm_values].mean(axis=1)
    client_a_df.insert(loc=client_a_df.columns.get_loc('salary_per_month_35ft') + 1, column='quasi_predicted_salary',
                       value=average_salary)

    raw_difference = client_a_df['salary_per_month_35ft'] - client_a_df['quasi_predicted_salary']
    client_a_df.insert(loc=client_a_df.columns.get_loc('quasi_predicted_salary') + 1, column='raw_salary_difference',
                       value=raw_difference)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    SD = scaler.fit_transform(client_a_df.raw_salary_difference.values.reshape(-1, 1))
    client_a_df.insert(loc=client_a_df.columns.get_loc('raw_salary_difference') + 1, column='SD',
                       value=SD.flatten())

    a = min(client_a_df.raw_salary_difference, key=abs)
    for value in client_a_df.raw_salary_difference:
        if value == a:
            print(client_a_df.index[client_a_df['raw_salary_difference'] == value].tolist())

    pay_label = []
    numeric_pay_label = []
    for value in client_a_df.SD:
        if value < -0.6:
            pay_label.append('underpaid')
            numeric_pay_label.append(-1)
        elif value > -0.45:
            pay_label.append('overpaid')
            numeric_pay_label.append(1)
        else:
            pay_label.append('fairly_paid')
            numeric_pay_label.append(0)

    client_a_df.insert(loc=client_a_df.columns.get_loc('raw_salary_difference') + 2, column='pay_label',
                       value=pay_label)
    client_a_df.insert(loc=client_a_df.columns.get_loc('raw_salary_difference') + 3, column='numeric_pay_label',
                       value=numeric_pay_label)

    return client_a_df
