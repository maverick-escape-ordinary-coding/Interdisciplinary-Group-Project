"""
Calculate Length of Service
"""

import pandas as pd


def service_float(df):
    try:
        service_lens = df['service_length']
        n = 'service_length'
    except:
        service_lens = df['Length_of_Service']
        n = 'Length_of_Service'
    new_service = []

    for item in service_lens:
        item = item.split(' ')
        new_item = round(float(int(item[0]) + int(item[2])/12), 2)
        new_service.append(new_item)
    df[n] = new_service
    return df