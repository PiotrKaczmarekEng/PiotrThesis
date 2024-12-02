README

This is the manual explaining the structure and usage of the model for the GFPSO supply chain. Firstly the necessary packages are denoted in (A), followed by an overview of the repository in (B), and finally the structure of the python code of the model in (C). The script was written using the Spyder IDE.


(A)
Make sure to install the necessary packages in the python environment:
pip install geopy
pip install numpy==1.16.6
pip install feedinlib==0.1.0rc4


(B)
The GitHub repository contains all the relevant files to the modelling of the GFPSO supply chain. The main components needed are found in folder "1. Main Model". The python script is contained in the "Model 3.1.py" file. The results output from the script will appear in the folder "Model Output". The input files required for the model include the weather data in the "Era 5 test data" folder, as well as the economic inputs in the "Parameter Data.xlsx" file.


(C)
The python script "Model 3.1.py" contains the relevant code for modelling the GFPSO supply chain. The script is split into 19 sections, described as follows:

1. Preamble
This section is used for importing all the relevant packages used throughout the rest of the script.

2. Excel parameters
This section imports the parameters from "Parameter Data.xlsx", as well as setting the other parameters used later in the functions and optimization. It is important to make sure that the time period data in the script matches the data in the "general" sheet in the excel file, otherwise errors may appear.

2.1 Sensitivity Analysis
If a sensitivity analysis is to be performed, the lines below 111 should be uncommented, and all lines below 204 indented. The "sensitivity_file_path" is currently being used for any outputs, not just sensitivity, and should be set to the location where the user wishes to see the model outputs. 

3. Wind Parameters
Here the wind costs are calculated in an array, with a function to compute the relevant cost matrix used in each optimization run.

4. Solar Power Function
Here the function for calculating the solar power output array is defined.

5. Wind Power Function
Here the function for calculating the wind turbine power output array is defined.

6. Transport Cost Functions & Parameters
A transport cost function is defined in order to retrieve the cost parameters of different modes, depending on location as input. The shipping costs are defined using the "Parameter Data.xlsx" file, while pipeline costs are calculated using functions of distance.

7. Location Selection
Here the study region is defined (based on the production location coordinates defined in Section 2). First the size has to be denoted, where the size represents how many discrete points are modelled in each row and column. For example, if size is 6, there will be 6x6 points, with 6 rows and 6 columns. This region can be offset using the start and end variables, and the resolution_map parameter determines the distance between studied locations. A map is plotted with each location in order to confirm the study region.

8. Prepare location loop
The model begins here. This loop will run the optimization model for each location, and medium-mode configuration defined previously.

9. Model Parameters and Sets
The Gurobi implementation begins here. The sets used in the model, as well as the final cost and technical parameters are defined in this section.

10. Variables
All the Gurobi variables are defined here.

11. Constraints
The Gurobi constraints are defined here.

12. Run Optimization
The Gurobi solver is used to optimze the objective function, with a loop for each timestep.

13. Post-processing
A dataframe is extracted from the optimization run results.

14. Extract Complete Output
A dataframe of all the outputs is extracted as "COMPLETE_OUTPUT". Additional dataframes of "df_full_all" and "df_full" are used for plotting later.

15. Plot Cost per kg Map 2050
A map of the cost density is plotted using plotly express, for the 2050 year (only if 2050 is in the studied time period). 

16. Plot Electrolyzer Map 2050
Similarly to the previous section, a plot of the electrolyzer number required is plotted on a map.

18. LCOH
The levelised cost of hydrogen is calculated for the complete time period, and exported to the folder "Model Outputs" under the selected path from "sensitivity_file_path".

19. Cost distribution
The cost distribution data is compiled into an excel file and exported to the folder "Model Outputs" under the selected path from "sensitivity_file_path".



