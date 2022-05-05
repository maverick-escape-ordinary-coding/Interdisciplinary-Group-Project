    """_summary_: Combine client datasets

    Returns:
        _type_: CSV
    """

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def IntersecOfSets(arr1, arr2, arr3, arr4):
    # Converting the arrays into sets
    s1 = set(arr1)
    s2 = set(arr2)
    s3 = set(arr3)
    s4 = set(arr4)

    set1 = s1.intersection(s2)
    set2 = set1.intersection(s3)
    result_set = set2.intersection(s4)
      
    # Converts resulting set to list
    final_list = list(result_set)
    print(final_list)

def IntersecOfSets3(arr1, arr2, arr3):
    # Converting the arrays into sets
    s1 = set(arr1)
    s2 = set(arr2)
    s3 = set(arr3)

    set1 = s1.intersection(s2)
    result_set = set1.intersection(s3)
      
    # Converts resulting set to list
    final_list = list(result_set)
    print(final_list)

client_A = pd.read_csv('cleaned_clientA.csv')
client_extra = pd.read_csv('cleaned_clientExtra.csv')
client_1_4 = pd.read_csv('cleaned_client1_4_ethnic.csv')
client_1_5 = pd.read_csv('cleaned_client1_5_ethnic.csv')

colsA = client_A.columns
print(colsA)

colsEx = client_extra.columns
print(colsEx)

cols1_4 = client_1_4.columns
print(cols1_4)

cols1_5 = client_1_5.columns
print(cols1_5)

IntersecOfSets3(colsA, cols1_4, cols1_5)

ClientA_join = client_A[['Division', 'City', 'Gender', 'Salary_Annual_35FT', 'Hourly_Equivalent']]

ClientA_join.rename(columns={'City' : 'Location', 'Salary_Annual_35FT' : 'Salary_Annual', 'Hourly_Equivalent' : 'Salary_Hourly'})

Client1_4_join = client_1_4[['Division', 'Location', 'Gender', 'Salary_Annual',	'Salary_Hourly']]

Client1_5_join = client_1_5[['Division', 'Location', 'Gender', 'SalaryAnnual', 'SalaryHourly']]

Client1_5_join.rename(columns={'SalaryAnnual' : 'Salary_Annual', 'SalaryHourly' : 'Salary_Hourly'})

join_clients = [ClientA_join, Client1_4_join, Client1_5_join]

join_clients.to_csv('clients_a_14_15_compilation.csv', index=False)
