    """_summary_: Preprocessing for Client Data Extra

    Returns:
        _type_: CSV
    """

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

client_extra = pd.read_csv('Extra Dataset_Public Sector_10k copy.csv')
client_extra.columns = client_extra.columns.str.replace(' ', '_') # replace spaces in column names

print('Starting size: ', client_extra.shape)


#Replace NANs where possible

# decided to NaNs in all allowance collumns although there were about 90% missing values, 
# but they may have an impact on the existence of outlayers or extreme values (e.g. overpaid people)
client_extra[['Z1_Allowance', 'Z2_Allowance', 'Flexi_Allowance', 'Flexi_Comms_Off', 'DoI_Flexi', 'Driver_Flexi', 
            'Market_Allowance', 'Shift_Disturbance_Allowance', 'Sat_Prem_x.5', 'Sun_Premium', 'Additional_Hours_P/T', 
            'Deputising_Allowance', 'Skill_Supplement', 'Typing_Proficiency', 'Overtime', 'Childcare', 'Service_Related_Pay', 
            'Location_Allowance', 'All_Included_Premium', 'All_Other_Included_Items', 'Flexibility_Allowance']] = client_extra[['Z1_Allowance', 
            'Z2_Allowance', 'Flexi_Allowance', 'Flexi_Comms_Off', 'DoI_Flexi', 'Driver_Flexi', 'Market_Allowance', 'Shift_Disturbance_Allowance', 
            'Sat_Prem_x.5', 'Sun_Premium', 'Additional_Hours_P/T', 'Deputising_Allowance', 'Skill_Supplement', 'Typing_Proficiency', 'Overtime', 
            'Childcare', 'Service_Related_Pay', 'Location_Allowance', 'All_Included_Premium', 'All_Other_Included_Items', 'Flexibility_Allowance']].fillna(0)

client_extra['Total_Allowances'] = client_extra['Z1_Allowance'] + client_extra['Z2_Allowance'] + client_extra['Flexi_Allowance'] + client_extra['Flexi_Comms_Off'] + \
                                    client_extra['DoI_Flexi'] + client_extra['Driver_Flexi'] + client_extra['Market_Allowance'] + client_extra['Shift_Disturbance_Allowance'] + \
                                    client_extra['Sat_Prem_x.5'] + client_extra['Sun_Premium'] + client_extra['Additional_Hours_P/T'] + client_extra['Deputising_Allowance'] + \
                                    client_extra['Skill_Supplement'] + client_extra['Typing_Proficiency'] + client_extra['Overtime'] + client_extra['Childcare'] + \
                                    client_extra['Service_Related_Pay'] + client_extra['Location_Allowance'] + client_extra['All_Included_Premium'] + \
                                    client_extra['All_Other_Included_Items'] + client_extra['Flexibility_Allowance']

#Add counted salary column assouming employee is working 35 hours FT their his wage

client_extra['Hourly_Pay'] = client_extra['Monthly_Pay_(basic)']/(4*client_extra['Actual_Worked_Hours'])

client_extra['Salary_per_week'] = client_extra['Hourly_Pay']*35
client_extra['Salary_per_month'] = (client_extra['Salary_per_week']*52)/12
client_extra['Salary_Annual_35FT'] = (client_extra['Salary_per_week']*52)


#Drop unnecessary columns

client_extra.drop(['UNIQUE_ID', 'Spec._Location_Allowance'], axis='columns', inplace=True)

client_extra = client_extra[client_extra.columns[client_extra.isnull().mean() < 0.4]]


# Delete NANs

client_extra = client_extra.dropna()


#Standardise columns —> 2DP

client_extra = client_extra.round(decimals=2)

print('Final size: ', client_extra.shape)

#Saving cleansed datasets

client_extra.to_csv('cleaned_clientExtra.csv', index=False)
client_extra[client_extra.columns.difference(['Ethnicity', 'Ethnic_Origin'])].to_csv('cleaned_clientExtra_without_ethnic.csv', index=False)
