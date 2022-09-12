import os
import numpy as np

import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

def remove(string):
    return string.replace(" ", "")

#Defining the euclidean distance
def euc(a, b):
    dist = np.sqrt(np.sum(np.square(a - b), axis=1))
    mean = dist.mean()
    return mean

lst_med_csv=[]
#Reading the csv file
#csv_name=['v0_10individuals_data.csv','v1_10individuals_data.csv']
#csv_name = ['v0_10individuals_data.csv','v1_10individuals_data.csv','v2_10individuals_data.csv','v3_10individuals_data.csv','v3_10individualsTESTING_data.csv','v3b_10individuals_data.csv','v4_10individuals_data.csv','v4b_10individuals_data.csv','v4test_10individuals_data.csv','v5_10individuals_data.csv','v5b_10individuals_data.csv','vfinal_10individuals_data.csv','vfinal_flipkey_10individuals_data.csv','vfinalLRnoflip_10individuals_data.csv','vfinalLRnoflip_flipkey_10individuals_data.csv','vfinalUnif_10individuals_data.csv']
csv_name = ['unique30_v0.csv','unique30_v1.csv','unique30_v2.csv','unique30_v3.csv','unique30_v3b.csv','unique30_v3TESTING.csv','unique30_v4.csv','unique30_v4b.csv','unique30_v4test.csv','unique30_v5.csv','unique30_v5b.csv','unique30_vfinal_flipkey.csv','unique30_vfinal.csv','unique30_vfinalLRnoflip_flipkey.csv','unique30_vfinalLRnoflip.csv','unique30_vfinalUnif.csv']
for csv in csv_name:
    path = '/home/abhinav1/projects/def-skrishna/shared/Unique_points_csvs/'
    Model_final = pd.read_csv(path+csv)
    #ID of the 10 individuals
    lst = ['01866',
     '02459',
     '01816',
     '03004',
     '03253',
     '01231',
     '00503',
     '02152',
     '02015',
     '01046']
    dicts={}
    lst_med=[]
    for pid in lst:
        a=pid
        Model_1= Model_final[Model_final['Image_ID'].str.startswith(a)]


        #Splitting the dataset first
        train, test = train_test_split(Model_1, test_size=0.3,shuffle=True)

        a_train=train['Model_Output']# taking a_train to be the [1,4] input
        b_train=train['GT_Value'] # taking a_train to be the [1,2] output

    #The values in the csv file are in string and they have to be converted into np array for performing SVR
    #First extracting all the values from the string(4 values for Model Output and 2 values for GT
        for i in range(0,len(a_train)):
            lis=[]
            a_train.iloc[i]=remove(a_train.iloc[i])
            x=a_train.iloc[i]
            a=x.split(',')
            x=a[0]
            x=x[9:]
            x1=a[1]

            x2=a[2]

            x3=a[3]
            x3=x3[0:-2]

            lis.append(float(x))
            lis.append(float(x1))
            lis.append(float(x2))
            lis.append(float(x3))
            a_train.iloc[i]=lis


        for i in range(0,len(b_train)):
            lis=[]
            x=b_train.iloc[i]
            a=x.split(',')
            x1=a[0]
            x1=x1[9:]
            x2=a[1]
            x2=x2[0:-2]
            lis.append(float(x1))
            lis.append(float(x2))
            b_train.iloc[i]=lis
    #Converting them into numpy array format

        for i in range(0,len(a_train)):
            a_train.iloc[i]=np.array(a_train.iloc[i])

        for i in range(0,len(b_train)):
            b_train.iloc[i]=np.array(b_train.iloc[i])
    #Transforming them into individual columns

        a_train = a_train.apply(pd.Series)
        b_train = b_train.apply(pd.Series)

        reg = MultiOutputRegressor(SVR())#Defining the SVR 
        #Defining the sweep parameters
        parameters = {'estimator__C': [20],  
                  'estimator__gamma': [0.06], 
                  'estimator__kernel': ['rbf'],
                 'estimator__epsilon':[0.01,0.1,1,10,100,1000]}  
        #Defining a custom scorer which will me Euclidean Distance
        custom_scorer = make_scorer(euc, greater_is_better=False)
        #Applying Grid Search
        grid_reg = GridSearchCV(estimator=reg,
                          param_grid=parameters,
                          scoring=custom_scorer,
                            cv=5,
                          verbose=3)
        #Fitting the model
        grid_reg.fit(a_train,b_train)

        best_para = grid_reg.best_params_#Finding the best parameter
        epsilon_SVR=best_para['estimator__epsilon']#Extracting epsilon value which has the lowest EUC
        #Defining the train and test inputs and outputs
        a_test=test['Model_Output']
        b_test=test['GT_Value']
        #Converting them into numpy arrays
        for i in range(0,len(a_test)):
            lis=[]
            a_test.iloc[i]=remove(a_test.iloc[i])
            x=a_test.iloc[i]
            a=x.split(',')
            x=a[0]
            x=x[9:]
            x1=a[1]

            x2=a[2]

            x3=a[3]
            x3=x3[0:-2]

            lis.append(float(x))
            lis.append(float(x1))
            lis.append(float(x2))
            lis.append(float(x3))
            a_test.iloc[i]=lis

        for i in range(0,len(b_test)):
            lis=[]
            x=b_test.iloc[i]
            a=x.split(',')
            x1=a[0]
            x1=x1[9:]
            x2=a[1]
            x2=x2[0:-2]
            lis.append(float(x1))
            lis.append(float(x2))
            b_test.iloc[i]=lis
      
        for i in range(0,len(a_test)):
            a_test.iloc[i]=np.array(a_test.iloc[i])
        for i in range(0,len(b_test)):
            b_test.iloc[i]=np.array(b_test.iloc[i])

        a_test = a_test.apply(pd.Series)
        b_test = b_test.apply(pd.Series)
        #Training SVR with the best SVR parameter
        reg = MultiOutputRegressor(SVR(C=20,epsilon=epsilon_SVR,gamma=0.06,kernel='rbf'))
        #Fitting the Model
        reg.fit(a_train,b_train)
        #Predicting the output
        y_pred = reg.predict(a_test)
        #Finding the loss
        loss=euc(y_pred,b_test)
        #Appending the loss in the list
        lst_med.append((pid,loss))
        
    lst_med_csv.append((csv,lst_med))
print(lst_med_csv)