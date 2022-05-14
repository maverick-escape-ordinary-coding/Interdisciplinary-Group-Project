import psycopg2
import pandas as pd
import numpy as np

import configparser

config = configparser.ConfigParser()
config.read('app.ini')
db_params = config['DB']

#load data from SQL
conn = psycopg2.connect(host='localhost', 
                       database=db_params['db'], 
                       user = db_params['user'], 
                       password = db_params['password'] )


cur = conn.cursor()
cur.execute("""SELECT table_name FROM information_schema.tables
       WHERE table_schema = 'public'""")
for table in cur.fetchall():
    print(table)
    
cur = conn.cursor()

def create_pandas_table(sql_query, database = conn):
    table = pd.read_sql_query(sql_query, database)
    return table

data = create_pandas_table("""SELECT * FROM client_ethnic_1_5""")

cur.close()
conn.close()

#for item in ('location', 'division', 'post_name', 'job_level', 'job_group'):
for item in ('location', 'division', 'post_name', 'job_level', 'job_group', 'category','age', 'years_service'):
    average_val = data.groupby(item, as_index=False)['annual_salary'].mean()
    average_val_dict = pd.Series(average_val.annual_salary.values, index=average_val[item]).to_dict()
    data.insert(loc=data.columns.get_loc(item) + 1, column='norm_salary_%s' % item,
                       value=[np.nan for i in range(data.shape[0])])
    data["norm_salary_%s" % item] = data[item].apply(lambda x: average_val_dict.get(x))
    
data.to_csv('client_1_5_as_model.csv', index=False) 

import seaborn as sns

ax = sns.boxplot(x="pay_label", y="SD", data=data, hue='gender', 
                 order=["underpaid", "fairly_paid", "overpaid"], palette="Set2")