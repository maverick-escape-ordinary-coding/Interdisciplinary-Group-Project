    """_summary_: Preprocessing for Client Data A

    Returns:
        _type_: CSV
    """

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#Client A

clientA = pd.read_csv('ClientA.csv')
clientA.columns = clientA.columns.str.replace(' ', '_') # replace spaces in column names
print('Starting size: ', clientA.shape)


#Remove duplicates

clientA = clientA.drop_duplicates()
clientA.shape # there's no duplicates, but just in case


#Check if UK only + city columns

uk_locations = ['Liverpool - St Pauls', 'Leeds - Bridgewater Place',
       'Manchester - Scott Place', 'RSA Liverpool', 'RSA Manchester',
       'London - 20 Fenchurch Street', 'Bristol - Redcliff Quay',
       'Newcastle - Orchard Street', 'Birmingham - Snow Hill',
       'Edinburgh - Fountain Bridge', 'Belfast - Queen Street',
       'Sheffield - Charter Row',  'Glasgow - Queen Street', "Dublin - George's Dock", 'London - Moor Place']

clientA = clientA.loc[clientA['Location'].isin(uk_locations)]

cities = []
for _, row in clientA.iterrows():
  if '-' in row['Location']:
    cities.append(row['Location'].split()[0])
  else:
    cities.append(row['Location'].split()[-1])

clientA['City'] = cities

print('Size after remove non- UK: ', clientA.shape)


#Replace NAN's where possible

clientA[['Ethnic_Origin', 'Nationality', 'Gender', 'Disabled', 'Marital_Status', 'Qualified_Solicitor', 
        'Benefit_Level', 'Post_Name', 'Career_Level', 'Role_Level', 'Tax', 'Net_Pay', 'Payroll', 'Total_Gross']] = clientA[['Ethnic_Origin', 'Nationality', 
        'Gender', 'Disabled', 'Marital_Status', 'Qualified_Solicitor', 'Benefit_Level', 'Post_Name', 'Career_Level', 'Role_Level', 'Tax', 'Net_Pay', 
        'Payroll', 'Total_Gross']].fillna('Unknown')

clientA['Total_Bonus_Amount'] = clientA['Total_Bonus_Amount'].fillna('No bonus')


#Add counted salary column assouming employee is working 35 hours FTE their his wage

clientA['Salary_per_week_35FT'] = clientA['Hourly_Equivalent']*35
clientA['Salary_per_month_35FT'] = (clientA['Salary_per_week_35FT']*52)/12
clientA['Salary_Annual_35FT'] = (clientA['Salary_per_week_35FT']*52)

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

pay_with_bonus = []
for _, row in clientA.iterrows():
  if isfloat(row['Total_Bonus_Amount']):
    pay_with_bonus.append(row['Salary_Annual_35FT'] + row['Total_Bonus_Amount'])
  else:
    pay_with_bonus.append('N/A')

clientA['SalaryPlusBonus_35FT'] = pay_with_bonus


#Drop unnecessary columns

clientA.drop(['Effective_Status', 'Payroll_Number', 'Manager_Payroll_Number', 'Pension_Level', 'EES_Pension'], axis='columns', inplace=True)

clientA = clientA[clientA.columns[clientA.isnull().mean() < 0.4]] # I checked which column will be removed, so I chose 40% of NaNs limit


#Standardise columns —> 2DP

clientA = clientA.round(decimals=2)


#Delete NANs

clientA = clientA.dropna()

print('Final size: ', clientA.shape)

#Saving cleansed datasets

clientA.to_csv('cleaned_clientA.csv', index=False)
clientA.loc[:, clientA.columns != 'Ethnic_Origin'].to_csv('cleaned_clientA_without_ethnic.csv', index=False)
clientA[clientA.columns.difference(['Tax', 'Net_Pay', 'Total_Gross'])].to_csv('cleaned_clientA_without_tax.csv', index=False)  # lot of unknown tax data, so maybe without will be needed
