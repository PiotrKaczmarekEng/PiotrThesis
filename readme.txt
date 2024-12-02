README

This is the manual explaining the structure and usage of the model for the GFPSO supply chain. Firstly the necessary packages are described in section (A), followed by an overview of the repository in section (B), and finally the structure of the python code of the model in section (C).

Section (A)
Make sure to install the packages necessary:
pip install geopy
pip install feedinlib==0.1.0rc4
pip install numpy==1.16.6

Section (B)
The repository contains all the relevant files to the modelling of the GFPSO supply chain. The main python script is contained in the "Model 2.0.py" file. The results output from the script will appear in the folder "csv_files". The input files required for the model include the weather data in the "NetCDF" folder, as well as the economic inputs in the "Inputdata econ.xlsx" file.


Section (C)
The python script "Model 2.0.py" contains the relevant code for modelling the GFPSO supply chain. The script is split into 22 sections, with each being described here.

1. Preamble
This section is used for importing all the relevant packages used throughout the rest of the script.

2. Excel parameters
This section imports the parameters from "Inputdata econ.xlsx", as well as setting the other parameters used later in the functions and optimization.

3. Wind Parameters
Here the wind costs are calculated in a matrix, with a function to compute the relevant cost matrix used in each optimization run.