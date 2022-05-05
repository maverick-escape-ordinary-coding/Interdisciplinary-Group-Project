    """_summary_
    Scrape cost of living data of cities in UK
    Returns:
        _type_: json files
    """

import urllib.request
from bs4 import BeautifulSoup
import json
import csv
base_url = 'https://www.numbeo.com/cost-of-living/in/' 
url_suffix = '-United-Kingdom'
rejection_string = 'Cannot'
cities = csv.DictReader(open('data/cities.csv'))
categories = {'Restaurants': 8, 'Markets': 19, 'Transportation': 8, 'Utilities (Monthly)': 3, 'Sports And Leisure': 8, 'Childcare': 2,'Clothing And Shoes': 4, 'Rent Per Month': 4 , 'Buy Apartment Price': 2, 'Salaries And Financing':2 }
value_labels = ['Type', 'Cost', 'Range']

#returns a soupified page if found
def get_page(city):
    page = urllib.request.urlopen(base_url + city)
    soup = BeautifulSoup(page, "lxml")
    if rejection_string in soup.find("title").text:
        page = urllib.request.urlopen(base_url + city + url_suffix)
        soup = BeautifulSoup(page, "lxml")
        if rejection_string in soup.find("title").text:
            return None
        else:
            return soup
    else:
        return soup

for row in cities:
    city = row['City']
    soup = get_page(city)
    city = city.lower()
    if (soup == None):
        print (city + ' Not Found')
    else:
        city_data = {city: []}
        for cat, num_subcats in categories.items():
            category = soup.find("div", text=cat)
            #get table rows below here up to the number specified for each category
            next = None
            cat_dict = {cat: []}
            for i in range(0,num_subcats):
                if next == None: 
                    next = category.find_next("tr")
                else:
                    next = next.find_next("tr")
                data = next.find_all("td")
                values = {}
                for j, value in enumerate(data):
                    value = value.text.strip()
                    if '£' in value:
                        value = value.replace(',','')
                        value = float(value.split()[0])
                    values[value_labels[j]] = value
                if len(values) > 0:
                    cat_dict[cat].append(values)
            city_data[city].append(cat_dict)

        with open('data/' + city.lower() + '.json','w') as file:
            file.write(json.dumps(city_data, indent = 4))

