#!/usr/bin/env python
# coding: utf-8

# __Objective__ =  Data Integration (ETL - Extract, Transform, Load) into Database
# 
# __version__ = _v4
# 
# __status__ = Done
# 
# __pending__ = Implement Class blueprint
# 
# __improvements__ = Column and Sequence Insertion optimisation
# 
# __run__ = python etl.py -p directory

# In[20]:


# import and install libraries if missing
import sys
import subprocess
import pkg_resources

required = {'psycopg2'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)


# In[21]:


import sys # Manipulating python runtime env 
import argparse # Handling arguments
import os # Handling file access
import pandas as pd # Data Manipulation and Analysis
import psycopg2 # Database Manipulation
from psycopg2 import Error # Database Connection Issue Exception
import configparser # Configuration Data for Database
import glob # handling file paths


# In[22]:


def connect_database(username,password,database):
    '''
    purpose: Function to establish connection to database and return an connection object
    input: username, password
    output: Database connection status, connection object
    '''
    try:
        
        # Connect to database
        connection = psycopg2.connect(user=username,
                                      password=password,
                                      host="",
                                      port="",
                                      database=database)

        # Create a cursor to perform database operations
        cursor = connection.cursor()
        
        # PostgreSQL details
        print("Database server information")
        print(connection.get_dsn_parameters(), "\n")
        
        # Executing a SQL query to fetch version info
        cursor.execute("SELECT version();")
        
        # Fetch result
        record = cursor.fetchone()
        print("You are connected to - ", record, "\n")
        
        return connection,cursor
    
    except (Exception, Error) as error:
        print("Error while connecting to Database", error)


# In[23]:


def rename_files(root_path):
    '''
    purpose: (Standardise) Rename file names in folders - Lowercase and join whitespaces
    input: path of the root folder
    '''
    try:
        if bool(glob.glob(root_path+'*')):
            print("Before:\n{0}".format(glob.glob(root_path+'*')))
            for path, folders, files in os.walk(root_path):

                for file_names in files:
                    os.rename(os.path.join(path, file_names), os.path.join(path, file_names.lower().replace(' ', '_')))
                for i in range(len(folders)):
                    new_file_name = folders[i].lower().replace(' ', '_')            
                    os.rename(os.path.join(path, folders[i]), os.path.join(path, new_file_name.lower()))
                    folders[i] = new_file_name

            print("\nAfter:\n{0}".format(glob.glob(root_path+'*')))
            return True
        else:
            print("Path doesn't exist!")
            return False
        
    except Exception as e:
        print("Problem encountered when renaming files: ", str(e))
        


# In[24]:


def extract_data(path, option):
    '''
    purpose: Extract cleaned data
    input: folder path (str)
    output: Cleaned Data (DataFrame)
    '''
    try:
        if rename_files(path):
            if option:
                if option != 'x':
                    # Construct file path and clean the file
                    if bool(glob.glob(os.path.join(path,'*_'+option+'.csv'))):
                        data = clean_data(''.join(glob.glob(os.path.join(path,'*_'+option+'.csv'))), option)
                        return data
                else:
                    config = configparser.ConfigParser()
                    config.read('app.ini')
                    db_params = config['DB']
                    app_params = config['APP']
                    
                    con,cursor = connect_database(db_params['user'],db_params['password'], db_params['db'])
                    
                    data = cur.execute("SELECT * from client_a")

                    # Fetching column names from DB
                    colnames = [desc[0] for desc in cur.description]
                    data = cur.fetchall()

                    # Converting database result into dataframe
                    inputData = pd.DataFrame(data,columns = colnames)     

                    # Close Database connection
                    if cur:
                        cur.close()
                        con.close()
                        pprint("Database connection is closed")
                    return inputData                    
                    
    
    except Exception as e:
        print("Problem encountered when extracting data: ", str(e))   
    


# In[25]:


def client_1_4(path):
    '''
    purpose: Clean data client_1.4
    input: folder path (str)
    output: Cleaned Data (DataFrame)
    '''    
    try:
        data = pd.read_csv(path)

        #Whole column is NaN
        data = data.drop('Start date of last position', 1)

        #Removed other rows that contain NaN 
        data = data.dropna()

        data_location = set()
        for Location in data['Location']:
            data_location.add(Location)
    #             print(data_location) #All locations are London 

        maternity = set()
        for Mat in data['Mat']:
            maternity.add(Mat)
    #             print(maternity) #All maternity leave is 0, disregard column

        data = data.drop('Mat', 1)

        #Rename columns with spaces
        data = data.rename({'PT?': 'Weekly_Hours', 'Hours PW': 'Hours_PW', 'Total allowances inc commission for Apr (per month)':'Total_Allowances'}, axis='columns')
        data = data.rename({'Commission (annual)': 'Commission_Annual', 'Bonus (annual)': 'Bonus_Annual', 'Total Bonus\n(Recipients\nonly)': 'Total_Bonus_and_Commision'}, axis='columns')
        data = data.rename({'Salary PA': 'Salary_Annual', 'Actual Salary (TP)': 'Salary_Monthly', 'PT?': 'PT'}, axis='columns')
        data = data.rename({'Support banding': 'Support_banding', 'Fee earner banding': 'Fee_earner_banding'}, axis='columns')
        data = data.rename({'Ethnic\nCategory': 'Ethnic_Category'}, axis='columns')
        

        #Matched columns, pick Weekly_Hours
        pd.Series(data.Weekly_Hours.values,index=data.Hours_PW).to_dict()

        #Calculate salary per hour and add a T/F column
        data['Salary_Hourly'] = data['Salary_Annual'] / (52*data['Weekly_Hours'])
        data['Commission_TF']  = (data['Commission_Annual'] > 0)
        data['Bonus_TF']  = (data['Bonus_Annual'] > 0)
        data = data.round(2)
        
        return data.iloc[:,[1,2,29,3,30,4,5,6,7,28,18,19,20,21,22,23,24,27,25]] #with ethnicity 

    except Exception as e:
        print("Problem encountered when cleaning data: ", str(e))
        


# In[26]:



def TF(row):
   '''
   purpose: Create a T/F column for the bonus columns
   '''
   if row["Bonus_NonVoucher"] > 0:
       return "True"
   elif row['Total_Bonus'] > 0:
       return "True" 
   else:
       return "False"

def client_1_5(path):
   '''
   purpose: Clean data client_1.5
   input: folder path (str)
   output: Cleaned Data (DataFrame)
   '''        
   try:
       data = pd.read_csv(path)

       #rename column for ease
       data = data.rename({"Continuous Service Date": 'length_service', "Birth Date": 'birth_date'}, axis='columns')
   
       if '/' not in str(data['length_service'][0]) or type(data['length_service'][0]) == 'int':
           #convert to date
           data['length_service'] = pd.to_datetime((data['length_service'] - 25569) * 86400.0, unit='s')
           data['birth_date'] = pd.to_datetime((data['birth_date'] - 25569) * 86400.0, unit='s')

       #create age and years service
       ref_date = dt.datetime.now()
       data['age'] = data['birth_date'].apply(lambda x: len(pd.date_range(start = x, end = ref_date, freq = 'Y'))) 
       data['years_service'] = data['length_service'].apply(lambda x: len(pd.date_range(start = x, end = ref_date, freq = 'Y'))) 

       data = data.drop('notes', 1) #All column NaNs

       employee = set()
       for Note in data['Relevant Employee']:
           employee.add(Note)

       #Already cleaned for relevant employees 
       data = data.drop('Relevant Employee', 1)
       data = data.drop('Qualification Date', 1)
       data

       Locations = set()
       for Location in data['Location']:
           Locations.add(Location)
   #     print(Locations)

       Locations2 = set()
       for Location2 in data['New Location']:
           Locations2.add(Location2)

       #All UK locations 
       #Only need one, as the cost of living is based on a city, pick Location

       data = data[data['Full-pay Relevant Employee']!= False]
       data

       #Rename columns with spaces 
       data = data.rename({'Hourly Pay': 'SalaryHourly', 'Bonus Without Voucher': 'Bonus_NonVoucher', 'Aggregate bonus':'Total_Bonus'}, axis='columns')
       data = data.rename({'Post Name': 'Post_Name', 'Job Level': 'Job_Level', 'Job Group':'Job_Group'}, axis='columns')
       data = data.rename({'Ethnic Origin': 'Ethnic_Origin', 'Ethnic Grouping': 'Ethnic_Grouping', 'White vs BAME': 'White_BAME'}, axis='columns')

       data['Bonus_TF'] = data.apply(TF, axis=1)

       data['SalaryWeekly'] = data['SalaryHourly']*35
       data['SalaryMonthly'] = (data['SalaryWeekly']*52)/12
       data['SalaryAnnual'] = (data['SalaryWeekly']*52)
       data['TotalPackage'] = data['SalaryAnnual'] + data['Total_Bonus']

       data = data.round(2)
       data = data.dropna()

       return data.iloc[:,[2,27,28,24,3,4,13,5,6,8,12,16,19,20,14,23,22,17,18,21]]
   
   except Exception as e:
       print("Problem encountered when cleaning 1_5 data: ", str(e))


# In[27]:


def isfloat(value):
    '''
    Check if value is of type float
    '''
    try:
        float(value)
        return True
    except ValueError:
        return False
            

def client_a(path):
    '''
    purpose: Clean data client_A
    input: folder path (str)
    output: Cleaned Data (DataFrame)
    '''        
    
    try:
        clientA = pd.read_csv(path)
        
        clientA.columns = clientA.columns.str.replace(' ', '_') # replace spaces in column names
#         print('Starting size: ', clientA.shape)


        """Remove duplicates"""

        clientA = clientA.drop_duplicates()
        clientA.shape # there's no duplicates, but just in case


        """Check if UK only + city columns"""

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


        """Replace NAN's where possible"""

        clientA[['Ethnic_Origin', 'Nationality', 'Gender', 'Disabled', 'Marital_Status', 'Qualified_Solicitor', 
                'Benefit_Level', 'Post_Name', 'Career_Level', 'Role_Level', 'Tax', 'Net_Pay', 'Payroll', 'Total_Gross']] = clientA[['Ethnic_Origin', 'Nationality', 
                'Gender', 'Disabled', 'Marital_Status', 'Qualified_Solicitor', 'Benefit_Level', 'Post_Name', 'Career_Level', 'Role_Level', 'Tax', 'Net_Pay', 
                'Payroll', 'Total_Gross']].fillna('Unknown')

        clientA['Total_Bonus_Amount'] = clientA['Total_Bonus_Amount'].fillna('No bonus')


        """Add counted salary column assouming employee is working 35 hours FTE their his wage"""

        clientA['Salary_per_week_35FT'] = clientA['Hourly_Equivalent']*35
        clientA['Salary_per_month_35FT'] = (clientA['Salary_per_week_35FT']*52)/12
        clientA['Salary_Annual_35FT'] = (clientA['Salary_per_week_35FT']*52)

        pay_with_bonus = []
        for _, row in clientA.iterrows():
            if isfloat(row['Total_Bonus_Amount']):
                pay_with_bonus.append(row['Salary_Annual_35FT'] + row['Total_Bonus_Amount'])
            else:
                pay_with_bonus.append('N/A')

        clientA['SalaryPlusBonus_35FT'] = pay_with_bonus


        """Drop unnecessary columns"""

        clientA.drop(['Effective_Status', 'Payroll_Number', 'Manager_Payroll_Number', 'Pension_Level', 'EES_Pension'], axis='columns', inplace=True)

        clientA = clientA[clientA.columns[clientA.isnull().mean() < 0.4]] # I checked which column will be removed, so I chose 40% of NaNs limit


        """Standardise columns —> 2DP"""

        clientA = clientA.round(decimals=2)


        """Delete NANs"""

        clientA = clientA.dropna()

#         print('Final size: ', clientA.shape)

        """Saving cleansed datasets"""
        
        return clientA

    except Exception as e:
        print("Problem encountered when cleaning client_A data: ", str(e))

    


# In[28]:


def client_extra(path):
    '''
    purpose: Clean data client_Extra
    input: folder path (str)
    output: Cleaned Data (DataFrame)
    '''        
    try:
        client_extra = pd.read_csv(path)

        client_extra.columns = client_extra.columns.str.replace(' ', '_') # replace spaces in column names



        """Replace NANs where possible"""

        # decided to NaNs in all allowance collumns although there were about 90% missing values, 
        # but they may have an impact on the existence of outlayers or extreme values (e.g. overpaid people)
        client_extra[['Z1_Allowance', 'Z2_Allowance', 'Flexi_Allowance', 'Flexi_Comms_Off', 'DoI_Flexi', 'Driver_Flexi', 
                    'Market_Allowance', 'Shift_Disturbance_Allowance', 'Sat_Prem_x.5', 'Sun_Premium', 'Additional_Hours_P/T', 
                    'Deputising_Allowance', 'Skill_Supplement', 'Typing_Proficiency', 'Overtime', 'Childcare', 'Service_Related_Pay', 
                    'Location_Allowance', 'All_Included_Premium', 'All_Other_Included_Items', 'Flexibility_Allowance']] = client_extra[['Z1_Allowance', 
                    'Z2_Allowance', 'Flexi_Allowance', 'Flexi_Comms_Off', 'DoI_Flexi', 'Driver_Flexi', 'Market_Allowance', 'Shift_Disturbance_Allowance', 
                    'Sat_Prem_x.5', 'Sun_Premium', 'Additional_Hours_P/T', 'Deputising_Allowance', 'Skill_Supplement', 'Typing_Proficiency', 'Overtime', 
                    'Childcare', 'Service_Related_Pay', 'Location_Allowance', 'All_Included_Premium', 'All_Other_Included_Items', 'Flexibility_Allowance']].fillna(0)

        client_extra['Total_Allowances'] = client_extra['Z1_Allowance'] + client_extra['Z2_Allowance'] + client_extra['Flexi_Allowance'] + client_extra['Flexi_Comms_Off'] +                                             client_extra['DoI_Flexi'] + client_extra['Driver_Flexi'] + client_extra['Market_Allowance'] + client_extra['Shift_Disturbance_Allowance'] +                                             client_extra['Sat_Prem_x.5'] + client_extra['Sun_Premium'] + client_extra['Additional_Hours_P/T'] + client_extra['Deputising_Allowance'] +                                             client_extra['Skill_Supplement'] + client_extra['Typing_Proficiency'] + client_extra['Overtime'] + client_extra['Childcare'] +                                             client_extra['Service_Related_Pay'] + client_extra['Location_Allowance'] + client_extra['All_Included_Premium'] +                                             client_extra['All_Other_Included_Items'] + client_extra['Flexibility_Allowance']

        """Add counted salary column assouming employee is working 35 hours FT their his wage"""

        client_extra['Hourly_Pay'] = client_extra['Monthly_Pay_(basic)']/(4*client_extra['Actual_Worked_Hours'])

        client_extra['Salary_per_week'] = client_extra['Hourly_Pay']*35
        client_extra['Salary_per_month'] = (client_extra['Salary_per_week']*52)/12
        client_extra['Salary_Annual_35FT'] = (client_extra['Salary_per_week']*52)


        """Drop unnecessary columns"""

        client_extra.drop(['UNIQUE_ID', 'Spec._Location_Allowance'], axis='columns', inplace=True)

        client_extra = client_extra[client_extra.columns[client_extra.isnull().mean() < 0.4]]


        """Delete NANs"""

        client_extra = client_extra.dropna()


        """Standardise columns —> 2DP"""

        client_extra = client_extra.round(decimals=2)

    #     print('Final size: ', client_extra.shape)

        return client_extra

    except Exception as e:
        print("Problem encountered when cleaning client_Extra data: ", str(e))
    
    


# In[29]:


def clean_data(path, option):
    '''
    Clean data based on the file options given
    '''
    try:
        if option == '1.4':
            return client_1_4(path)
        if option == '1.5':
            return client_1_5(path)
        if option == 'a':
            return client_a(path)
        if option == 'extra':
            return client_extra(path)
        if option == 'x':
            return client_x(path)        
        
    except Exception as e:
        print("Problem encountered when cleaning data: ", str(e))
            
    
    


# In[30]:


def service_float(df):
    '''
    converting length of service for an year
    '''
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

def transform_data(data,option):
    '''
    purpose: Change columns names and create primary column 
    input: dataframe (data), string (file option)
    output: dataframe
    '''
    try:
        if option == '1.4':
            data.columns = ['total_allowances',
                            'commission_annual',
                            'commission_tf',
                            'bonus_annual',
                            'bonus_tf',
                            'total_bonus_commission',
                            'weekly_hours',
                            'annual_salary',
                            'monthly_salary',
                            'hourly_salary',
                            'division',
                            'team',
                            'location',
                            'category',
                            'support_banding',
                            'fee_earner_banding',
                            'gender',
                            'age',
                            'ethnic_category']

        if option == '1.5':
            data.columns = ['hourly_salary',
                            'annual_salary',
                            'total_package',
                            'bonus_tf',
                            'bonus_non_voucher',
                            'total_bonus',
                            'full_time_equivalent',
                            'gender',
                            'location',
                            'division',
                            'category',
                            'post_name',
                            'job_level',
                            'job_group',
                            'length_service',
                            'years_service',
                            'age',
                            'ethnic_origin',
                            'ethnic_grouping',
                            'white_bame']

        if option == 'a':
            data.columns = ['division',
                             'groups',
                             'area',
                             'team',
                             'location',
                             'job_title',
                             'status',
                             'category',
                             'join_date',
                             'company_tenure',
                             'service_length',
                             'pay_amount',
                             'hourly_equivalent',
                             'notice_rule',
                             'ethnic_origin',
                             'nationality',
                             'gender',
                             'disabled',
                             'work_style',
                             'hours_Per_Week',
                             'days_per_week',
                             'contract',
                             'appointment_type',
                             'marital_status',
                             'qualified_solicitor',
                             'career_level',
                             'role_level',
                             'benefit_level',
                             'full_time_equivalent',
                             'currency',
                             'ft_hours',
                             'ft_pay_amount',
                             'wp_description',
                             'career_effective_date',
                             'career_change_reason',
                             'pay_from_date',
                             'pay_change_reason',
                             'non_executive_director',
                             'payroll_name',
                             'tax',
                             'net_pay',
                             'basic_pay_nl_uk',
                             'payroll',
                             'post_name',
                             'arc_pension',
                             'auto_daily_rate',
                             'auto_hourly_rate',
                             'ees_nic',
                             'total_gross',
                             'total_bonus_amount',
                             'city',
                             'salary_per_week_35ft',
                             'salary_per_month_35ft',
                             'salary_annual_35ft',
                             'salary_plus_bonus_35ft']

        if option == 'extra':
            data.columns = ['main_band_range',
                            'rank_band',
                            'pay_point',
                            'monthly_pay_basic',
                            'gender',
                            'disability',
                            'age',
                            'actual_worked_hours',
                            'full_time_equivalent',
                            'ft_pt',
                            'service_start_date',
                            'length_of_service',
                            'ethnicity',
                            'ethnic_origin',
                            'z1_allowance',
                            'z2_allowance',
                            'flexi_allowance',
                            'flexi_comms_off',
                            'doi_flexi',
                            'driver_flexi',
                            'market_allowance',
                            'shift_disturbance_allowance',
                            'sat_prem_x_5',
                            'sun_premium',
                            'additional_hours_p_t',
                            'deputising_allowance',
                            'skill_supplement',
                            'typing_proficiency',
                            'overtime',
                            'childcare',
                            'service_related_pay',
                            'location_allowance',
                            'all_included_premium',
                            'all_other_included_items',
                            'flexibility_allowance',
                            'annualised_salaries_fte',
                            'total_allowances',
                            'hourly_pay',
                            'salary_per_week',
                            'salary_per_month',
                            'salary_annual_35_ft'] 
            
        if option == 'x':

            data = data[['division', 'groups', 'area', 'team', 'job_title', 'category', 'company_tenure', 'hourly_equivalent',
                    'notice_rule', 'gender', 'work_style', 'qualified_solicitor', 'appointment_type', 'career_level',
                    'role_level', 'ft_pay_amount', 'post_name', 'benefit_level', 'total_bonus_amount', 'city', 'service_length']]

            for item in ['division', 'groups', 'area', 'team', 'job_title', 'category', 'post_name', 'benefit_level']:
                uniq_vals = pd.unique(df[item])
                anonym_vals = ['{}_{}'.format(item, i) for i in range(1, len(uniq_vals)+1)]
                replace_dict = {uniq_vals[j]: anonym_vals[j] for j in range(len(uniq_vals))}
                data = data.replace({item: replace_dict}) 

            data = service_float(data)            

        # Creating Employee column with index values    
        data.reset_index(inplace=True,drop=True)
        data.index += 1
        data['employee'] = data.index.values.tolist()                


        return data

    except Exception as e:
        print('Problem encountered when cleaning data: ', str(e))
        

    
    
    


# In[31]:


def load_data(data,option):
    '''
    purpose: Transfer data to database
    input: dataframe (data), string (file option)
    output: Database Record insertion/updation status   
    '''
    
    # Fetch Config data for database connection
    config = configparser.ConfigParser()
    config.read('app.ini')
    db_params = config['DB']
    app_params = config['APP']
    
    con,cursor = connect_database(db_params['user'],db_params['password'], db_params['db'])

    val_to_insert = data.values.tolist()
    
    if option == '1.4':
        table_columns = '''INSERT INTO client_ethnic_1_4(employee,total_allowances,
                                        commission_annual,
                                        commission_tf,
                                        bonus_annual,
                                        bonus_tf,
                                        total_bonus_commission,
                                        weekly_hours,
                                        annual_salary,
                                        monthly_salary,
                                        hourly_salary,
                                        division,
                                        team,
                                        location,
                                        category,
                                        support_banding,
                                        fee_earner_banding,
                                        gender,
                                        age,
                                        ethnic_category) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                        %s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (employee) DO UPDATE
                                        SET total_allowances = excluded.total_allowances,
                                        commission_annual = excluded.commission_annual,
                                        commission_tf = excluded.commission_tf,
                                        bonus_annual = excluded.bonus_annual,
                                        bonus_tf = excluded.bonus_tf,
                                        total_bonus_commission = excluded.total_bonus_commission,
                                        weekly_hours = excluded.weekly_hours,
                                        annual_salary = excluded.annual_salary,
                                        monthly_salary = excluded.monthly_salary,
                                        hourly_salary = excluded.hourly_salary,
                                        division = excluded.division,
                                        team = excluded.team,
                                        location = excluded.location,
                                        category = excluded.category,
                                        support_banding = excluded.support_banding,
                                        fee_earner_banding = excluded.fee_earner_banding,
                                        gender = excluded.gender,
                                        age = excluded.age,
                                        ethnic_category = excluded.ethnic_category;
                                        '''

    if option == '1.5':        
        table_columns = '''INSERT INTO client_ethnic_1_5(employee,hourly_salary,
                                        annual_salary,
                                        total_package,
                                        bonus_tf,
                                        bonus_non_voucher,
                                        total_bonus,
                                        full_time_equivalent,
                                        gender,
                                        location,
                                        division,
                                        category,
                                        post_name,
                                        job_level,
                                        job_group,
                                        length_service,
                                        years_service,
                                        age,
                                        ethnic_origin,
                                        ethnic_grouping,
                                        white_bame) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (employee) DO UPDATE
                                        SET hourly_salary = excluded.hourly_salary,
                                        annual_salary = excluded.annual_salary,
                                        total_package = excluded.total_package,
                                        bonus_tf = excluded.bonus_tf,
                                        bonus_non_voucher = excluded.bonus_non_voucher,
                                        total_bonus = excluded.total_bonus,
                                        full_time_equivalent = excluded.full_time_equivalent,
                                        gender = excluded.gender,
                                        location = excluded.location,
                                        division = excluded.division,
                                        category = excluded.category,
                                        post_name = excluded.post_name,
                                        job_level = excluded.job_level,
                                        job_group = excluded.job_group,
                                        length_service = excluded.length_service,
                                        years_service = excluded.years_service,
                                        age = excluded.age,
                                        ethnic_origin = excluded.ethnic_origin,
                                        ethnic_grouping = excluded.ethnic_grouping,
                                        white_bame = excluded.white_bame;
                                        '''

    if option == 'a':
        table_columns = '''INSERT INTO client_ethnic_a(employee,division,
                                         groups,
                                         area,
                                         team,
                                         location,
                                         job_title,
                                         status,
                                         category,
                                         join_date,
                                         company_tenure,
                                         service_length,
                                         pay_amount,
                                         hourly_equivalent,
                                         notice_rule,
                                         ethnic_origin,
                                         nationality,
                                         gender,
                                         disabled,
                                         work_style,
                                         hours_Per_Week,
                                         days_per_week,
                                         contract,
                                         appointment_type,
                                         marital_status,
                                         qualified_solicitor,
                                         career_level,
                                         role_level,
                                         benefit_level,
                                         full_time_equivalent,
                                         currency,
                                         ft_hours,
                                         ft_pay_amount,
                                         wp_description,
                                         career_effective_date,
                                         career_change_reason,
                                         pay_from_date,
                                         pay_change_reason,
                                         non_executive_director,
                                         payroll_name,
                                         tax,
                                         net_pay,
                                         basic_pay_nl_uk,
                                         payroll,
                                         post_name,
                                         arc_pension,
                                         auto_daily_rate,
                                         auto_hourly_rate,
                                         ees_nic,
                                         total_gross,
                                         total_bonus_amount,
                                         city,
                                         salary_per_week_35ft,
                                         salary_per_month_35ft,
                                         salary_annual_35ft,
                                         salary_plus_bonus_35ft) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                         %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                         %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                         %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                         %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                         %s,%s,%s,%s,%s) ON CONFLICT (employee) DO UPDATE
                                        SET division = excluded.division,
                                         groups = excluded.groups,
                                         area = excluded.area,
                                         team = excluded.team,
                                         location = excluded.location,
                                         job_title = excluded.job_title,
                                         status = excluded.status,
                                         category = excluded.category,
                                         join_date = excluded.join_date,
                                         company_tenure = excluded.company_tenure,
                                         service_length = excluded.service_length,
                                         pay_amount = excluded.pay_amount,
                                         hourly_equivalent = excluded.hourly_equivalent,
                                         notice_rule = excluded.notice_rule,
                                         ethnic_origin = excluded.ethnic_origin,
                                         nationality = excluded.nationality,
                                         gender = excluded.gender,
                                         disabled = excluded.disabled,
                                         work_style = excluded.work_style,
                                         hours_Per_Week = excluded.hours_Per_Week,
                                         days_per_week = excluded.days_per_week,
                                         contract = excluded.contract,
                                         appointment_type = excluded.appointment_type,
                                         marital_status = excluded.marital_status,
                                         qualified_solicitor = excluded.qualified_solicitor,
                                         career_level = excluded.career_level,
                                         role_level = excluded.role_level,
                                         benefit_level = excluded.benefit_level,
                                         full_time_equivalent = excluded.full_time_equivalent,
                                         currency = excluded.currency,
                                         ft_hours = excluded.ft_hours,
                                         ft_pay_amount = excluded.ft_pay_amount,
                                         wp_description = excluded.wp_description,
                                         career_effective_date = excluded.career_effective_date,
                                         career_change_reason = excluded.career_change_reason,
                                         pay_from_date = excluded.pay_from_date,
                                         pay_change_reason = excluded.pay_change_reason,
                                         non_executive_director = excluded.non_executive_director,
                                         payroll_name = excluded.payroll_name,
                                         tax = excluded.tax,
                                         net_pay = excluded.net_pay,
                                         basic_pay_nl_uk = excluded.basic_pay_nl_uk,
                                         payroll = excluded.payroll,
                                         post_name = excluded.post_name,
                                         arc_pension = excluded.arc_pension,
                                         auto_daily_rate = excluded.auto_daily_rate,
                                         auto_hourly_rate = excluded.auto_hourly_rate,
                                         ees_nic = excluded.ees_nic,
                                         total_gross = excluded.total_gross,
                                         total_bonus_amount = excluded.total_bonus_amount,
                                         city = excluded.city,
                                         salary_per_week_35ft = excluded.salary_per_week_35ft,
                                         salary_per_month_35ft = excluded.salary_per_month_35ft,
                                         salary_annual_35ft = excluded.salary_annual_35ft,
                                         salary_plus_bonus_35ft = excluded.salary_plus_bonus_35ft
                                         
                                         '''
        
    if option == 'extra':
        table_columns = '''INSERT INTO client_ethnic_extra(employee,main_band_range,
                                rank_band,
                                pay_point,
                                monthly_pay_basic,
                                gender,
                                disability,
                                age,
                                actual_worked_hours,
                                full_time_equivalent,
                                ft_pt,
                                service_start_date,
                                length_of_service,
                                ethnicity,
                                ethnic_origin,
                                z1_allowance,
                                z2_allowance,
                                flexi_allowance,
                                flexi_comms_off,
                                doi_flexi,
                                driver_flexi,
                                market_allowance,
                                shift_disturbance_allowance,
                                sat_prem_x_5,
                                sun_premium,
                                additional_hours_p_t,
                                deputising_allowance,
                                skill_supplement,
                                typing_proficiency,
                                overtime,
                                childcare,
                                service_related_pay,
                                location_allowance,
                                all_included_premium,
                                all_other_included_items,
                                flexibility_allowance,
                                annualised_salaries_fte,
                                total_allowances,
                                hourly_pay,
                                salary_per_week,
                                salary_per_month,
                                salary_annual_35_ft) VALUES (%s,%s,%s,%s,%s,
                                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) 
                                ON CONFLICT (employee) DO UPDATE
                                SET main_band_range = excluded.main_band_range,
                                rank_band = excluded.rank_band,
                                pay_point = excluded.pay_point,
                                monthly_pay_basic = excluded.monthly_pay_basic,
                                gender = excluded.gender,
                                disability = excluded.disability,
                                age = excluded.age,
                                actual_worked_hours = excluded.actual_worked_hours,
                                full_time_equivalent = excluded.full_time_equivalent,
                                ft_pt = excluded.ft_pt,
                                service_start_date = excluded.service_start_date,
                                length_of_service = excluded.length_of_service,
                                ethnicity = excluded.ethnicity,
                                ethnic_origin = excluded.ethnic_origin,
                                z1_allowance = excluded.z1_allowance,
                                z2_allowance = excluded.z2_allowance,
                                flexi_allowance = excluded.flexi_allowance,
                                flexi_comms_off = excluded.flexi_comms_off,
                                doi_flexi = excluded.doi_flexi,
                                driver_flexi = excluded.driver_flexi,
                                market_allowance = excluded.market_allowance,
                                shift_disturbance_allowance = excluded.shift_disturbance_allowance,
                                sat_prem_x_5 = excluded.sat_prem_x_5,
                                sun_premium = excluded.sun_premium,
                                additional_hours_p_t = excluded.additional_hours_p_t,
                                deputising_allowance = excluded.deputising_allowance,
                                skill_supplement = excluded.skill_supplement,
                                typing_proficiency = excluded.typing_proficiency,
                                overtime = excluded.overtime,
                                childcare = excluded.childcare,
                                service_related_pay = excluded.service_related_pay,
                                location_allowance = excluded.location_allowance,
                                all_included_premium = excluded.all_included_premium,
                                all_other_included_items = excluded.all_other_included_items,
                                flexibility_allowance = excluded.flexibility_allowance,
                                annualised_salaries_fte = excluded.annualised_salaries_fte,
                                total_allowances = excluded.total_allowances,
                                hourly_pay = excluded.hourly_pay,
                                salary_per_week = excluded.salary_per_week,
                                salary_per_month = excluded.salary_per_month,
                                salary_annual_35_ft = excluded.salary_annual_35_ft
                                
                                '''      
        
    if option == 'x':
        table_columns = '''INSERT INTO client_x(division, 
        groups, area, team, job_title, category, 
        company_tenure, hourly_equivalent,
        notice_rule, gender, work_style, qualified_solicitor, 
        appointment_type, career_level,
        role_level, ft_pay_amount, post_name, benefit_level, 
        total_bonus_amount, city, service_length,employee
                                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                         %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                         %s,%s,%s) ON CONFLICT (employee) DO UPDATE
                                        SET division = excluded.division,
                                         groups = excluded.groups,
                                         area = excluded.area,
                                         team = excluded.team,
                                         job_title = excluded.job_title,
                                         work_style = excluded.work_style,
                                         qualified_solicitor = excluded.qualified_solicitor,
                                         category = excluded.category,
                                         company_tenure = excluded.company_tenure,
                                         service_length = excluded.service_length,
                                         hourly_equivalent = excluded.hourly_equivalent,
                                         notice_rule = excluded.notice_rule,
                                         gender = excluded.gender,
                                         appointment_type = excluded.appointment_type,
                                         career_level = excluded.career_level,
                                         role_level = excluded.role_level,
                                         benefit_level = excluded.benefit_level,
                                         total_bonus_amount = excluded.total_bonus_amount,
                                         full_time_equivalent = excluded.full_time_equivalent,
                                         ft_pay_amount = excluded.ft_pay_amount,
                                         post_name = excluded.post_name,
                                         city = excluded.city
                                         
                                         '''
            

    try:
        cursor.executemany(table_columns,val_to_insert)

        con.commit()
        print("Record Inserted Successfully")

    except (Exception, psycopg2.Error) as e:
        print("Record insertion failed {}".format(e))

    finally:
        con.close()
        cursor.close()   

    


# In[ ]:


def main(args):
    '''
    purpose: Point of execution
    input: path (argument)
    '''
    try:
        # Parse Arguments
        parser = argparse.ArgumentParser(description='Please input directory path to your source files')
        parser.add_argument("-p",'--path', required=True)
        args = parser.parse_args(args)
        path = args.path
        
        # Process the files based on the options given
        if ('file_options' in app_params):
            num_features = app_params['file_options'].split(',')
            options = [s.strip() for s in num_features]
            
            if options:
                for each_option in options:
                    data = extract_data(path,each_option) 
                    data = transform_data(data, each_option)
                    load_data(data,each_option)
            
    except Exception:
        print("ETL Failed!")
        
if __name__ == "__main__":
    main(sys.argv[1:])
        

