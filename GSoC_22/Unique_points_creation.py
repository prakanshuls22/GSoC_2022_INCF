import numpy as np
import pandas as pd
import random

Main_Collection = pd.read_csv('/home/sheenu22/projects/def-skrishna/shared/v1_10individuals_data.csv')

IDs = ['01866',
 '02459',
 '01816',
 '03004',
 '03253',
 '01231',
 '00503',
 '02152',
 '02015',
 '01046']

data = []

random.seed(22)

for x in IDs:
    part_ind = Main_Collection[Main_Collection['Image_ID'].str.startswith(x)]
    
    main_dict = {}
    
    for ind in part_ind.index:
        if part_ind['GT_Value'][ind] not in main_dict.keys():
            main_dict[part_ind['GT_Value'][ind]] = [ind]
        else:
            main_dict[part_ind['GT_Value'][ind]].append(ind)
    
    temp_list = []
    
    for i,x in enumerate(main_dict):
        temp_list.append(random.choice(main_dict[x]))
        
    for x in temp_list:
        data.append([part_ind['Image_ID'][x],part_ind['Model_Output'][x],part_ind['GT_Value'][x]])
        
df=pd.DataFrame(data,columns=['Image_ID','Model_Output','GT_Value'])
path_csv = '/home/sheenu22/projects/def-skrishna/shared/Unique_points_csvs/unique30_v1.csv'
df.to_csv(path_csv, index = False)