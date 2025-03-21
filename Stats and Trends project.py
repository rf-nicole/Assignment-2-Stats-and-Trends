# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 21:29:33 2025

@author: ropaf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    """
    Function will read the data in Worldbank format 
    process the data into two data frames

    Parameters
    ----------
    filename : str
        DESCRIPTION.

    Returns
    -------
    df_years_columns: pandas dataframe
    df_countries_columns : pandas dataframe

    """
    df = pd.read_csv(filename)
    print("Columns in the dataset:", df.columns)  # Check the column names
    print(df.head(20))  # Inspect the first 20 rows of the dataset
    df.columns = df.columns.astype(str)
    years_to_remove = [str(year) for year in range(1960, 1990)]  # List of years to drop
    df = df.drop(columns=years_to_remove, errors='ignore')  # Drop years with missing data
    # Drop rows or columns with NaN values
    df_years_columns = df.dropna(subset=df.columns.difference(['Country Name']), axis=0, how='any')
    df_countries_columns = df.set_index('Country Name').transpose()
    
    
    # Check if 'Country Name' is in the index
    print("Countries in the dataset:", df_years_columns.index.tolist()[:20])
    
    return df_years_columns, df_countries_columns

CO2_emissions = 'API_EN.GHG.CO2.IP.MT.CE.AR5_DS2_en_csv_v2_14525.csv'
CO2_emissions_years, CO2_emissions_countries = read_data(CO2_emissions)

ren_energy_use = 'API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_4123.csv'
energy_use_years, energy_use_countries = read_data(ren_energy_use)

print('The columns are: ', CO2_emissions_countries.columns)
print(CO2_emissions_years.describe())
print(CO2_emissions_countries.describe())
print(energy_use_years.describe())
print(energy_use_countries.describe())

selected_countries = ['United States', 'Germany', 'India', 'Brazil', 'Zimbabwe', 'Afghanistan']
CO2_selected = CO2_emissions_countries[selected_countries]
ren_energy_selected = energy_use_countries[selected_countries]

CO2_selected =CO2_selected.apply(pd.to_numeric, errors ='coerce')
ren_energy_selected = ren_energy_selected.apply(pd.to_numeric, errors='coerce')

CO2_selected_clean = CO2_selected.dropna()
ren_energy_selected_clean = ren_energy_selected.dropna()


#focusing on the most recent 20 years
recent_years = [str(year) for year in range(2000, 2021)]
CO2_recent = CO2_selected.loc[recent_years]
ren_energy_recent = ren_energy_selected.loc[recent_years]

#focusing from 1990-1999 since our renewable enrgy data starts from 1989
first_years = [str(year) for year in range (1990, 2000)]
CO2_beginning = CO2_selected.loc[first_years]
ren_energy_beginning = ren_energy_selected.loc[first_years]

#removing NaN values
CO2_recent_clean = CO2_recent.dropna()
ren_energy_recent_clean = ren_energy_recent.dropna()
CO2_beginning_clean = CO2_beginning.dropna()
ren_energy_beginning_clean = ren_energy_beginning.dropna()

#finding the correlation
recent_corr = CO2_recent_clean.corrwith(ren_energy_recent_clean)
print('Correlation of CO2 emissions with Renewable Energy Use for the most recent years:', recent_corr)
beginning_corr = CO2_beginning_clean.corrwith(ren_energy_beginning_clean)
print('Correlation for 1990-1999 is: ', beginning_corr)


#plotting the data
plt.figure(figsize=(12,8))
for country in selected_countries:
    plt.plot(CO2_selected.index, CO2_selected[country], label= f'{country}-CO2 Emissions', linestyle='--')   
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('CO2 Emissions Over Time')
plt.legend(title='Countries')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,8))
for country in selected_countries:
    plt.plot(ren_energy_selected.index, ren_energy_selected[country], label=f'{country}-Renewable Energy Use', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Renewable Energy Use over Time')
plt.legend(title='Countries')
plt.xticks(rotation=45)
plt.show()


