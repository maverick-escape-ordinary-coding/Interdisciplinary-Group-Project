    """_summary_: Preprocessing for Client Data 1.4

    Returns:
        _type_: CSV
    """

import pandas as pd
data = pd.read_csv('Client1_4.csv')
data

list(data.columns)
data = data.drop('Start date of last position', 1)

#Removed other rows that contain NaN 
data = data.dropna()
data

data_location = set()
for Location in data['Location']:
    data_location.add(Location)
print(data_location) #All locations are London 

maternity = set()
for Mat in data['Mat']:
    maternity.add(Mat)
print(maternity) #All maternity leave is 0, disregard column

data = data.drop('Mat', 1)

#Rename columns with spaces
data = data.rename({'PT?': 'Weekly_Hours', 'Hours PW': 'Hours_PW', 'Total allowances inc commission for Apr (per month)':'Total_Allowances'}, axis='columns')
data = data.rename({'Commission (annual)': 'Commission_Annual', 'Bonus (annual)': 'Bonus_Annual', 'Total Bonus\n(Recipients\nonly)': 'Total_Bonus_and_Commision'}, axis='columns')
data = data.rename({'Salary PA': 'Salary_Annual', 'Actual Salary (TP)': 'Salary_Monthly', 'PT?': 'PT'}, axis='columns')
data = data.rename({'Support banding': 'Support_banding', 'Fee earner banding': 'Fee_earner_banding'}, axis='columns')
data = data.rename({'Ethnic\nCategory': 'Ethnic_Category'}, axis='columns')
print(data.columns)

#Matched columns, pick Weekly_Hours
pd.Series(data.Weekly_Hours.values,index=data.Hours_PW).to_dict()

#Calculate salary per hour and add a T/F column
data['Salary_Hourly'] = data['Salary_Annual'] / (52*data['Weekly_Hours'])
data['Commission_TF']  = (data['Commission_Annual'] > 0)
data['Bonus_TF']  = (data['Bonus_Annual'] > 0)
data = data.round(2)

#Selected neccessary columns and reorder
X = data.iloc[:,[1,2,29,3,30,4,5,6,7,28,18,19,20,21,22,23,24,27,25]] #with ethnicity 
Y = data.iloc[:,[1,2,29,3,30,4,5,6,7,28,18,19,20,21,22,23,24,27]] 
X.to_csv('cleaned_client1_4_ethnic.csv', index=False) 
Y.to_csv('cleaned_client1_4.csv', index=False) 

