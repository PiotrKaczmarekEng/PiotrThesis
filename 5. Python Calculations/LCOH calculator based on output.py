# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:17:17 2024

@author: spide
"""

E = 3

data_file = os.getcwd() 
data_file = os.path.dirname(os.path.realpath('__file__')) + '\csv_files\Try 11 (fixed depth, fixed lat lon functions)\Output_j'+str(E+1)+'.xlsx'
wb = load_workbook(data_file,data_only=True) # creating workbook


demand = 121425.74257425741
timestep = 3


wb[str(l*timestep+2020)].cell(row=2, column=3).value    # cost per kg year 2020 location 1 (row = location, column = costperkg, sheet = year)


LCOH_df = pd.DataFrame(columns=['longitude','latitude','LCOH'],index=[list(range(size))])



npvprod_list = []
npvcost_list = []
prodh2 = demand*1000
counter = 0
counter3 = 0
for vert in range(size):
    counter2=0
    for j in range(size):
        for l in L:
            year_prod = l*timestep
            dr = 1/((1+DiscountRate)**year_prod)
            # Costperkg_ = COMPLETE_OUTPUT[counter3].loc['Costs per kg hydrogen (euros)',l*timestep+2020]
            Costperkg_ = wb[str(l*timestep+2020)].cell(row=counter3+2, column=3).value
            TotalYearlyCost = Costperkg_*prodh2
            npvcost_list.append(dr*TotalYearlyCost)
            npvprod_list.append(dr*prodh2)
            
            print(counter3)
            print(l*timestep+2020)
            print(Costperkg_)
            
        npvcost = sum(npvcost_list)
        npvcost_list = []
        npvprod = sum(npvprod_list)        
        npvprod_list = []

        LCOH = npvcost / npvprod
        LCOH_df.loc[counter3,'LCOH'] = LCOH
        LCOH_df.loc[counter3,'longitude'] = loc_matrix[vert][counter2][0]
        LCOH_df.loc[counter3,'latitude'] = loc_matrix[vert][counter2][1]
        counter+=1
        counter2+=1
        counter3+=1

min_index = LCOH_df['LCOH'].idxmin()
optimal_location_df = LCOH_df.iloc[min_index[0]]
optimal_location = "("+str(optimal_location_df['latitude'])+", "+str(optimal_location_df['longitude'])+")"

LCOH_df['Optimal'] = (LCOH_df['latitude'] == optimal_location_df['latitude']) & (LCOH_df['longitude'] == optimal_location_df['longitude'])


# create a excel writer object
with pd.ExcelWriter(os.path.dirname(os.path.realpath('__file__')) +"\csv_files\Try 11 (fixed depth, fixed lat lon functions)\LCOH_j"+str(E+1)+".xlsx") as writer:
   
    # use to_excel function and specify the sheet_name and index 
    # to store the dataframe in specified sheet
    
    LCOH_df.to_excel(writer, sheet_name='j='+str(E+1), index=False)