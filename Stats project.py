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
    df=df.dropna(axis=0, how='all', subset=df.columns.difference(['Country Name']))
    df_years_columns = df.set_index('Country Name')
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

#grouping by region
if 'Country Code' in CO2_emissions_years.columns:
    CO2_by_region = CO2_emissions_years.groupby('Country Code')
    print(CO2_by_region.head())

selected_countries = ['United States', 'Germany', 'India', 'Brazil', 'Zimbabwe', 'Afghanistan']

CO2_selected = CO2_emissions_countries[selected_countries].apply(pd.to_numeric, errors ='coerce')
ren_energy_selected = energy_use_countries[selected_countries].apply(pd.to_numeric, errors='coerce')


#focusing on the most recent 20 years
recent_years = [str(year) for year in range(2000, 2021)]
CO2_recent = CO2_selected.loc[recent_years].dropna()
ren_energy_recent = ren_energy_selected.loc[recent_years].dropna()

#focusing from 1990-1999 since our renewable enrgy data starts from 1989
first_years = [str(year) for year in range (1990, 2000)]
CO2_beginning = CO2_selected.loc[first_years].dropna()
ren_energy_beginning = ren_energy_selected.loc[first_years].dropna()


#finding the correlation
recent_corr = CO2_recent.corrwith(ren_energy_recent)
print('Correlation of CO2 emissions with Renewable Energy Use for the most recent years:', recent_corr)
beginning_corr = CO2_beginning.corrwith(ren_energy_beginning)
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

print(CO2_selected.columns)

year_1990 = '1990'
year_2020 = '2020'

# Extract data for both years
CO2_1990 = CO2_selected.loc[year_1990]
ren_energy_1990 = ren_energy_selected.loc[year_1990]

CO2_2020 = CO2_selected.loc[year_2020]
ren_energy_2020 = ren_energy_selected.loc[year_2020]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

x = np.arange(len(selected_countries))
width = 0.4  # Width of bars

# Plot for 1990
axes[0].bar(x - width/2, CO2_1990, width, label='CO₂ Emissions (Mt)', color='red')
axes[0].bar(x + width/2, ren_energy_1990, width, label='Renewable Energy Use (%)', color='green')
axes[0].set_title('CO₂ Emissions & Renewable Energy (1990)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(selected_countries, rotation=45)
axes[0].legend()

# Plot for 2020
axes[1].bar(x - width/2, CO2_2020, width, label='CO₂ Emissions (Mt)', color='red')
axes[1].bar(x + width/2, ren_energy_2020, width, label='Renewable Energy Use (%)', color='green')
axes[1].set_title('CO₂ Emissions & Renewable Energy (2020)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(selected_countries, rotation=45)
axes[1].legend()

# Set common y-axis label
fig.supylabel('Values')

# Show the plot
plt.show()

def ave_rate_of_change(data, years_range):
    """
    Calculating the average rate of change over a given range of years

    Parameters
    ----------
    data : pandas DataFrame
        Data for country
    years_range : list
        the years used to calculate the rate of change

    Returns
    -------
    rate_of_change: pandas Series
    Average rate of change for each country in the given ranges

    """
    start_year, end_year = str(years_range[0]), str(years_range[-1])
    rate_of_change = (data.loc[end_year] - data.loc[start_year]) / (len(years_range) - 1)
    return rate_of_change

#1990-1999
first_years = [str(year) for year in range(1990,2000)]
CO2_first_rate_of_change = ave_rate_of_change(CO2_selected, first_years)
energy_first_rate_of_change = ave_rate_of_change(ren_energy_selected, first_years)
print('Average rate of change for CO2 Emissions (1990-1999): ', CO2_first_rate_of_change)
print('Average rate of change for Renewable Energy Use (1990-1999): ', energy_first_rate_of_change)

recent_years = [str(year) for year in range(2000,2021)]
CO2_recent_rate_of_change = ave_rate_of_change(CO2_selected, recent_years)
energy_recent_rate_of_change = ave_rate_of_change(ren_energy_selected, recent_years)
print('Average rate of change for CO2 emissions (2000-2020): ', CO2_recent_rate_of_change)
print('Average rate of change for renewable energy use (1990-1999): ', energy_recent_rate_of_change)

def moving_average_plot(rate_of_change, window=3, title='Moving average rate of change'):
    """
    Visualises the rate of change as a moving average

    Parameters
    ----------
    rate_of_change : pandas Series
    Rate of change for each country 

    """
    num_countries = len(rate_of_change.columns)
    fig, axes = plt.subplots(num_countries, 1, figsize=(10,5*num_countries), sharex=True)
    if num_countries ==1:
        axes=[axes]
    for ax, country in zip(axes, rate_of_change.columns):
        moving_avg = rate_of_change[country].rolling(window=window).mean()
        ax.plot(moving_avg.index, moving_avg, linestyle='-', color='b', label= country)
        ax.set_title(f'{country} - {title}')
        ax.set_ylabel('Rate of change')
        ax.legend()
        ax.grid(True)
        
    plt.xlabel('Year')
    plt.tight_layout()
    plt.show()

moving_average_plot(CO2_beginning, window=3, title='C02 Emissions 1990-1999')
moving_average_plot(ren_energy_beginning, window=3, title='Renewable Energy use 1990-1999')
moving_average_plot(CO2_recent, window=3, title='CO2 emissions 2000-2020')
moving_average_plot(ren_energy_recent, window=3, title='Renewable Energy Use 2000-2020')

#performing the bootstrapping
from stats import bootstrap
def bootstrap_correlation(data1, data2, confidence_level=0.95, nboot=10000):
    """ bootstrap confidence interval for the correlation"""
    corrs=[]
    n=len(data1)
    
    for i in range(nboot):
        indices = np.random.choice(n, n, replace=True)
        sample_corr = np.corrcoef(data1.iloc[indices], data2.iloc[indices])[0,1]
        corrs.append(sample_corr)
    
    low,high = bootstrap(np.array(corrs), np.mean, confidence_level, nboot)
    return low, high

low_bound, high_bound = bootstrap_correlation(CO2_recent['United States'], ren_energy_recent['United States'])
print('95% Confidence Interval for Correlation (US, 2000-2020): ', low_bound, high_bound)

#check to see if correlation is significant
if low_bound > 0 or high_bound < 0:
    print('The correlation is statistically significant.')
else:
    print('The correlation is NOT statistically significant.')
    


