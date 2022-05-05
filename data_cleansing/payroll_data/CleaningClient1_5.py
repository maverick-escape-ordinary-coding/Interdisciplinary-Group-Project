    """_summary_: Preprocessing for Client Data 1.5

    Returns:
        _type_: CSV
    """

import pandas as pd
import datetime as dt
data = pd.read_csv("Client1_5.csv")
data

#rename column for ease
data = data.rename({"Continuous Service Date": 'length_service', "Birth Date": 'birth_date'}, axis='columns')

#convert to date
data['length_service'] = pd.to_datetime((data['length_service'] - 25569) * 86400.0, unit='s')
data['birth_date'] = pd.to_datetime((data['birth_date'] - 25569) * 86400.0, unit='s')

#create age and years service
ref_date = dt.datetime.now()
data['age'] = data['birth_date'].apply(lambda x: len(pd.date_range(start = x, end = ref_date, freq = 'Y'))) 
data['years_service'] = data['length_service'].apply(lambda x: len(pd.date_range(start = x, end = ref_date, freq = 'Y'))) 

list(data.columns)

data = data.drop('notes', 1) #All column NaNs

data = data.drop('Relevant Employee', 1)
data = data.drop('Qualification Date', 1)

data = data[data['Full-pay Relevant Employee']!= False]
data

#Rename columns with spaces 
data = data.rename({'Hourly Pay': 'SalaryHourly', 'Bonus Without Voucher': 'Bonus_NonVoucher', 'Aggregate bonus':'Total_Bonus'}, axis='columns')
data = data.rename({'Post Name': 'Post_Name', 'Job Level': 'Job_Level', 'Job Group':'Job_Group'}, axis='columns')
data = data.rename({'Ethnic Origin': 'Ethnic_Origin', 'Ethnic Grouping': 'Ethnic_Grouping', 'White vs BAME': 'White_BAME'}, axis='columns')

#Create a T/F column for the bonus columns 
def TF(row):
    if row["Bonus_NonVoucher"] > 0:
        return "True"
    elif row['Total_Bonus'] > 0:
        return "True" 
    else:
        return "False"

data['Bonus_TF'] = data.apply(TF, axis=1)

data['SalaryWeekly'] = data['SalaryHourly']*35
data['SalaryMonthly'] = (data['SalaryWeekly']*52)/12
data['SalaryAnnual'] = (data['SalaryWeekly']*52)
data['TotalPackage'] = data['SalaryAnnual'] + data['Total_Bonus']

data = data.round(2)
data = data.dropna()
print(data)

X = data.iloc[:,[2,27,28,24,3,4,13,5,6,8,12,16,19,20,14,23,22,17,18,21]]
Y = data.iloc[:,[2,27,28,24,3,4,13,5,6,8,12,16,19,20,14,23,22]]
X.to_csv('cleaned_client1_5_ethnic.csv', index=False) 
Y.to_csv('cleaned_client1_5.csv', index=False) 




