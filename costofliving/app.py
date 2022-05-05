import pandas as pd
import json
from flask import Flask
from flask import render_template

app = Flask(__name__)

inclusion_keys_consumer = ['Restaurants', 'Markets', 'Clothing And Shoes', 'Transportation',  'Sports And Leisure', 'Utilities (Monthly)', 'Childcare']
inclusion_keys_rent = ['Rent Per Month']

keys = {'consumer': inclusion_keys_consumer, 'rent': inclusion_keys_rent, 'combined': inclusion_keys_consumer.append(inclusion_keys_rent)}

@app.route('/data/<city_name>')
@app.route('/data/<city_name>/<type>')
def get_city_data_json(city_name, type='consumer'):
    try:
        data = get_city_data(city_name, type).to_json()
    except:
        raise Exception("%s:%s not found" % (city_name, type))
    return data

#returns a dataframe with a summary of the cost of living data for a given city/type of costs
def get_city_data(city_name, type):
    city_name = city_name.lower()
    file_data = json.loads(open('data/' + city_name.lower() + '.json').read())
    categorized_data = file_data[city_name]
    city_costs = []
    for category in categorized_data:
        category_name = next(iter(category))
        if (category_name in keys[type]):
            for cost in category[category_name]:
                cost['Category'] = category_name
                city_costs.append(cost)
    df = pd.DataFrame(city_costs)
    df = df.rename({'Range': city_name + ' Range', 'Cost': city_name + " Cost"}, axis=1) 
    return df

#returns a data frame merged from two cities               
def get_city_comparison_data(city_1, city_2, type = 'consumer'):
    try:
        df_city_1 = get_city_data(city_1, type)
    except:
        raise Exception("%s:%s not found" % (city_1, type))
    
    try:
        df_city_2 = get_city_data(city_2, type)
    except:
        raise Exception("%s:%s not found" % (city_2, type))
    
    df = pd.merge(df_city_1,df_city_2, on=["Category", "Type"])
    df['% Diff'] = ((df.iloc[:,1]/df.iloc[:,4])*100)-100
    return df

@app.route('/compare/<city1>/<city2>')
@app.route('/compare/<city1>/<city2>/<type>')
def compare(city1, city2, type = 'consumer'):
    try:
        data = get_city_comparison_data(city1, city2, type)['% Diff']
    except Exception as e:
        return {'error' : str(e)}
    return {
        'City 1': city1,
        'City 2': city2,
        '% Diff': data.mean()
    } 
@app.route('/compare_html/<city1>/<city2>')
def compare_html(city1, city2):
    try:
        data = get_city_comparison_data(city1, city2)['% Diff']
    except Exception as e:
        return {'error' : str(e)}
    
    data = data.mean()
    
    return render_template('root.html', city_1 = city1, city_2 = city2, data=data, title = 'COL')
    