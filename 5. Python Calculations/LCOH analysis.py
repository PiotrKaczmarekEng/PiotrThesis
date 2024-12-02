# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:02:16 2024

@author: spide
"""


import pandas as pd

# Load the Excel file into a DataFrame
file_path = os.getcwd() 
file_path = os.path.dirname(os.path.realpath('__file__')) + '\csv_files\Try 3\Output_j2.xlsx'
# df = pd.read_excel(file_path, sheet_name='LCOH')

# [df_j1, df_j2, df_j3, df_j4]

file_path = os.getcwd() 
file_path = os.path.dirname(os.path.realpath('__file__')) + '\csv_files\Try 3\Output_j1.xlsx'

df_j1 = pd.read_excel(
    file_path, 
    sheet_name='LCOH', 
    usecols='A:F',  # Columns A to E
    skiprows=1,     # Skip the first row to start from A2
    nrows=17        # Read the next 17 rows to cover the range A2:E18
)

file_path = os.getcwd() 
file_path = os.path.dirname(os.path.realpath('__file__')) + '\csv_files\Try 3\Output_j2.xlsx'

df_j2 = pd.read_excel(
    file_path, 
    sheet_name='LCOH', 
    usecols='A:F',  # Columns A to E
    skiprows=1,     # Skip the first row to start from A2
    nrows=17        # Read the next 17 rows to cover the range A2:E18
)

file_path = os.getcwd() 
file_path = os.path.dirname(os.path.realpath('__file__')) + '\csv_files\Try 3\Output_j3.xlsx'

df_j3 = pd.read_excel(
    file_path, 
    sheet_name='LCOH', 
    usecols='A:F',  # Columns A to E
    skiprows=1,     # Skip the first row to start from A2
    nrows=17        # Read the next 17 rows to cover the range A2:E18
)

file_path = os.getcwd() 
file_path = os.path.dirname(os.path.realpath('__file__')) + '\csv_files\Try 3\Output_j4.xlsx'

df_j4 = pd.read_excel(
    file_path, 
    sheet_name='LCOH', 
    usecols='A:F',  # Columns A to E
    skiprows=1,     # Skip the first row to start from A2
    nrows=17        # Read the next 17 rows to cover the range A2:E18
)

df_J = [df_j1, df_j2, df_j3, df_j4]

# df_j1 = pd.read_excel(
#     file_path, 
#     sheet_name='LCOH', 
#     usecols='A:E',  # Columns A to E
#     skiprows=1,     # Skip the first row to start from A2
#     nrows=17        # Read the next 17 rows to cover the range A2:E18
# )

for j in range(4):
# Print the first five rows of the DataFrame
    print(df_J[j].head())
