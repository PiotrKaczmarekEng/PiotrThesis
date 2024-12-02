# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:12:42 2024

@author: spide
"""

import pandas as pd
import os

# List of Excel file paths to be combined
data_file = os.getcwd() 

file_path = 'csv_files/Sensitivity Analysis/Lower Case/'

excel_files = [] # ['file1.xlsx', 'file2.xlsx', 'file3.xlsx']  # Add your file paths here

SSP_string_list = ["SSP_A1",
"SSP_A2",
"SSP_B1",
"SSP_B2",
"SSP_B3",
"SSP_B4",
"SSP_C1",
"SSP_C2",
"SSP_TC"]

for i in range(len(SSP_string_list)):
    
    new_file_path = file_path + SSP_string_list[i] + 'LCOH_j4.xlsx'
    print(new_file_path)
    excel_files.append(new_file_path)

# Create a new Excel writer object
with pd.ExcelWriter('csv_files/Sensitivity Analysis/Lower Case/LCOH_j4.xlsx', engine='openpyxl') as writer:
    # Iterate over each Excel file
    for file in excel_files:
        # Extract the file name without extension for sheet name
        sheet_name = SSP_string_list[excel_files.index(file)]
        
        # Read the current Excel file
        df = pd.read_excel(file)
        
        # Write the data to a new sheet in the combined Excel file
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Save the combined Excel file
    # writer.save()

print("Files combined successfully!")

#%%

import pandas as pd

# List of Excel file paths and identifiers

data_file = os.getcwd() 

file_path = 'csv_files/Sensitivity Analysis/Reference Case/Combined'

excel_files = {
    'csv_files/Sensitivity Analysis/Reference Case/Combined/LCOH_j1.xlsx': 'j1',
    'csv_files/Sensitivity Analysis/Reference Case/Combined/LCOH_j2.xlsx': 'j2',
    'csv_files/Sensitivity Analysis/Reference Case/Combined/LCOH_j3.xlsx': 'j3',
    'csv_files/Sensitivity Analysis/Reference Case/Combined/LCOH_j4.xlsx': 'j4'
}

# List of sheet names (common across all files)
sheet_names = ['SSP_A1', 'SSP_A2', 'SSP_B1', 'SSP_B2', 'SSP_B3', 'SSP_B4', 'SSP_C1', 'SSP_C2', 'SSP_TC']

# Create a new Excel writer object for the combined file
with pd.ExcelWriter('csv_files/Sensitivity Analysis/Reference Case/Combined/combined_LCOH.xlsx', engine='openpyxl') as writer:
    
    # Iterate over each sheet
    for sheet in sheet_names:
        combined_data = pd.DataFrame()  # Initialize an empty DataFrame for each sheet
        
        # Iterate over each Excel file
        for file, identifier in excel_files.items():
            # Read the current sheet from the current file
            df = pd.read_excel(file, sheet_name=sheet)
            
            # Add a new column 'Source' to indicate the identifier (e.g., 'j1', 'j2', etc.)
            df['Source'] = identifier
            
            # Append the data to the combined DataFrame
            combined_data = pd.concat([combined_data, df], ignore_index=True)
        
        # Write the combined data to the corresponding sheet in the output file
        combined_data.to_excel(writer, sheet_name=sheet, index=False)
    
    # Save the combined Excel file
    # writer.save()

print("Files combined successfully!")

