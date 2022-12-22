from flask import Flask, jsonify
from flask import make_response
from flask import request

app = Flask(__name__)

# For Model
import pandas
import csv
import os
import time
import subprocess
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# np.set_printoptions(threshold=np.inf)
# pd.set_option("max_rows", None)


class user_dietPlan_model():
    def user_dietPlan_method(self, data):

        DiseaseCategory = request.args.get("DiseaseCategory")
        BreakfastCalories = request.args.get("BreakfastCalories")
        LunchCalories = request.args.get("LunchCalories")
        SnackCalories = request.args.get("SnackCalories")
        DinnerCalories = request.args.get("DinnerCalories")
        bmi = request.args.get("bmi")   
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        df=pandas.read_csv(os.path.join(BASE_DIR ,"data/foodDiabetesChanged.csv"))
        # print(df)

        df.isnull().sum()

        df['Protein'] = df['Protein'].interpolate()
        df['Carbs'] = df['Carbs'].interpolate()
        df['Fat'] = df['Fat'].interpolate()
        df['Fiber'] = df['Fiber'].interpolate()

        df.isnull().sum()

        df.duplicated().sum()

        df.isin([np.inf,-np.inf]).sum()

        pd.isna(df).sum()

        df.info()

        df['Protein'] = df['Protein'].astype(float)
        df['Carbs'] = df['Carbs'].astype(float)
        df['Fat'] = df['Fat'].astype(float)
        df['Fiber'] = df['Fiber'].astype(float)
        df['Calories'] = df['Calories'].astype(float)

        sns.heatmap(df.isnull())

        ################################# Random Forest Algorithm ############################################

        pd.get_dummies(df['VegNonveg']).head(189)

        Veg = pd.get_dummies(df['VegNonveg'],drop_first=True)

        new_data = pd.concat([df,Veg], axis=1)

        new_data.drop('VegNonveg',axis=1, inplace=True)

        #Label Encoding
        from sklearn import preprocessing
        from sklearn.preprocessing import OrdinalEncoder

        label_encoder = preprocessing.LabelEncoder()
        enc =  OrdinalEncoder()
        label_encoder = preprocessing.LabelEncoder()
        new_data[["Measurings"]] = enc.fit_transform(new_data[["Measurings"]])
        new_data['Food_items']=label_encoder.fit_transform(df['Food_items'])

        Prediabtic=new_data['PreDiabetic']
        Diabetic=new_data['Diabetic']

        IsPatient=[]
        for i in range(len(Prediabtic)):
            if Prediabtic[i]==1 and Diabetic[i] == 1:
                IsPatient.append(2)
            if Prediabtic[i]==0 and Diabetic[i] == 0:
                IsPatient.append(0)
            if Prediabtic[i]==1 and Diabetic[i] == 0:
                IsPatient.append(0)
            if Prediabtic[i]==0 and Diabetic[i] == 1:
                IsPatient.append(1)    
        
        new_data['IsPatient'] = IsPatient
        new_data.drop(['PreDiabetic','Diabetic'],axis=1,inplace=True);
        Y = new_data.IsPatient
        new_data.drop('IsPatient',inplace=True, axis=1)
        X=new_data
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0)

        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=100,criterion='gini',random_state=0)
        classifier.fit(X_train, y_train)

        ###################################### Testing Code ######################################

        df_test = pandas.read_csv("data/TestfoodDiabetesChanged.csv")

        df_test['Protein'] = df_test['Protein'].interpolate()
        df_test['Carbs'] = df_test['Carbs'].interpolate()
        df_test['Fat'] = df_test['Fat'].interpolate()
        df_test['Fiber'] = df_test['Fiber'].interpolate()

        pd.get_dummies(df_test['VegNonveg'])
        Veg_test = pd.get_dummies(df_test['VegNonveg'],drop_first=True)
        df_test = pd.concat([df_test,Veg_test], axis=1)
        df_test.drop('VegNonveg',axis=1, inplace=True)

        #Label Encoding
        from sklearn import preprocessing
        from sklearn.preprocessing import OrdinalEncoder

        label_encoder = preprocessing.LabelEncoder()
        enc =  OrdinalEncoder()
        df_test[["Measurings"]] = enc.fit_transform(df_test[["Measurings"]])
        df_test['Food_items']=label_encoder.fit_transform(df_test['Food_items'])

        Prediabtic_test=df_test['PreDiabetic']
        Diabetic_test=df_test['Diabetic']

        isPatient=[]
        for i in range(len(Prediabtic_test)):
                if Prediabtic_test[i]==1 and Diabetic_test[i] == 1:
                    isPatient.append(2)
                if Prediabtic_test[i]==0 and Diabetic_test[i] == 0:
                    isPatient.append(0)
                if Prediabtic_test[i]==1 and Diabetic_test[i] == 0:
                    isPatient.append(0)
                if Prediabtic_test[i]==0 and Diabetic_test[i] == 1:
                    isPatient.append(1)                    

        df_test['IsPatient'] = isPatient
        df_test.drop(['PreDiabetic','Diabetic'],axis=1,inplace=True);
        Y_test = df_test.IsPatient
        df_test.drop('IsPatient',inplace=True, axis=1)
        X_test=df_test
        ''' Instead of X there should be X_test '''
        y_pred=classifier.predict(X_test)

        from sklearn.metrics import classification_report,accuracy_score
        '''Instead of Y there will be y_test '''
        print(accuracy_score(Y_test,y_pred))

        ############################ Separating Lunch, Dinner etc Items A/c to user's Condition ####################

        # DiseaseCategory = "Prediabetic"            #---------------------

        dataID=[]
        for i in range(len(y_pred)):
            if DiseaseCategory == "Prediabetic":
                if y_pred[i] == 0:
                    dataID.append(i)
                if y_pred[i] == 2:
                    dataID.append(i)
            if DiseaseCategory == "Diabetic":
                if y_pred[i] == 1:
                    dataID.append(i)
                if y_pred[i] == 2:
                    dataID.append(i)

        newdf= df_test.iloc[dataID]

        Breakfastfoodseparated=[]
        Lunchfoodseparated=[]
        Dinnerfoodseparated=[]
        Snackfoodseparated=[]
                
        BreakfastfoodseparatedID=[]
        LunchfoodseparatedID=[]
        DinnerfoodseparatedID=[]
        SnackfoodseparatedID=[]

        Breakfastdata=newdf['Breakfast']
        BreakfastdataNumpy=Breakfastdata.to_numpy()
                                   
        Lunchdata=newdf['Lunch']
        LunchdataNumpy=Lunchdata.to_numpy()
                                                    
        Dinnerdata=newdf['Dinner']
        DinnerdataNumpy=Dinnerdata.to_numpy()

        Snackdata=newdf['Snack']
        SnackdataNumpy=Snackdata.to_numpy()

        Food_itemsdata=newdf['Food_items']

        for i in range(len(Breakfastdata)):
            if BreakfastdataNumpy[i]==1:
                BreakfastfoodseparatedID.append(i)
            if LunchdataNumpy[i]==1:
                LunchfoodseparatedID.append(i)
            if DinnerdataNumpy[i]==1:
                DinnerfoodseparatedID.append(i)
            if SnackdataNumpy[i]==1:
                SnackfoodseparatedID.append(i)

        ########### Retreiving Data by Loc Method ##################

        # retrieving Lunch data rows by loc method 
        LunchfoodseparatedIDdata = newdf.iloc[LunchfoodseparatedID]

        # retrieving Breakfast data rows by loc method 
        BreakfastfoodseparatedIDdata = newdf.iloc[BreakfastfoodseparatedID]

        # retrieving Snack data rows by loc method 
        SnackfoodseparatedIDdata = newdf.iloc[SnackfoodseparatedID]

        # retrieving Dinner data rows by loc method 
        DinnerfoodseparatedIDdata = newdf.iloc[DinnerfoodseparatedID]

        ############################################# K Mean Algorithm ###################################################

        ################# Breakfast #################

        km = KMeans(n_clusters=3)
        y_predict = km.fit_predict(BreakfastfoodseparatedIDdata[['Calories','type']])
        BreakfastfoodseparatedIDdata['cluster'] = y_predict

        BreakfastfoodseparatedIDdata['Food_items'] = label_encoder.inverse_transform(BreakfastfoodseparatedIDdata['Food_items'])
        BreakfastfoodseparatedIDdata[["Measurings"]] = enc.inverse_transform(BreakfastfoodseparatedIDdata[["Measurings"]])  

        k_rng=range(1,12)

        sse=[]

        for k in k_rng:
            km = KMeans(n_clusters=k)
            km.fit(BreakfastfoodseparatedIDdata[['type','Calories']])
            sse.append(km.inertia_)

        #################### Lunch ###################

        km = KMeans(n_clusters=3)

        ly_predict = km.fit_predict(LunchfoodseparatedIDdata[['Calories','type']])

        LunchfoodseparatedIDdata['cluster'] = ly_predict

        LunchfoodseparatedIDdata['Food_items'] = label_encoder.inverse_transform(LunchfoodseparatedIDdata['Food_items'])
        LunchfoodseparatedIDdata[["Measurings"]] = enc.inverse_transform(LunchfoodseparatedIDdata[["Measurings"]])

        ##################### Snack ####################

        km = KMeans(n_clusters=3)

        sy_predict = km.fit_predict(SnackfoodseparatedIDdata[['Calories','type']])

        SnackfoodseparatedIDdata['cluster'] = sy_predict

        SnackfoodseparatedIDdata['Food_items'] = label_encoder.inverse_transform(SnackfoodseparatedIDdata['Food_items'])
        SnackfoodseparatedIDdata[["Measurings"]] = enc.inverse_transform(SnackfoodseparatedIDdata[["Measurings"]])

        ##################### Dinner ####################

        km = KMeans(n_clusters=3)
        
        dy_predict = km.fit_predict(DinnerfoodseparatedIDdata[['Calories','type']])
        
        DinnerfoodseparatedIDdata['cluster'] = dy_predict

        DinnerfoodseparatedIDdata['Food_items'] = label_encoder.inverse_transform(DinnerfoodseparatedIDdata['Food_items'])
        DinnerfoodseparatedIDdata[["Measurings"]] = enc.inverse_transform(DinnerfoodseparatedIDdata[["Measurings"]])    


        ############################################# Package Creation ################################################

        ### Here we will intialize BMI variable ###
        # bmi = 18  #--------------------------------------

        bmi = float(bmi)
        if (bmi < 16.0):
                bmiinfo="According to your BMI, you are Severely Underweight"
                clbmi=1
        elif ( bmi >= 16.0 and bmi < 18.5):
                bmiinfo="According to your BMI, you are Underweight"
                clbmi=1
        elif ( bmi >= 18.5 and bmi < 24.9):
                bmiinfo="According to your BMI, you are Healthy"
                clbmi=1
        elif ( bmi >= 25.0 and bmi < 29.9):
                bmiinfo="According to your BMI, you are Overweight"
                clbmi=0
        elif ( bmi >=30.0):
                bmiinfo="According to your BMI, you are Severely Overweight"
                clbmi=0

        ##################### Breakfast Package ####################
        
        ### Here we will intialize Breakfast Calories variable ###

        BreakfastPackage=[]

        BreakfastPackageID=[]


        BreakfastCaloriesCount = 0.0
        BreakfastProteinCount = 0.0
        BreakfastCarbsCount = 0.0
        BreakfastFatCount = 0.0
        BreakfastFiberCount = 0.0

        # BreakfastCalories = 349      #--------------------
        BreakfastCalories = int(BreakfastCalories)
        #------------------------

        Btype = BreakfastfoodseparatedIDdata.iloc[BreakfastfoodseparatedIDdata['Calories'].argmin()]
        Bclu = Btype['cluster']


        bmicl = 0

        if(bmicl == 0):
            BreakfastItem = BreakfastfoodseparatedIDdata.loc[BreakfastfoodseparatedIDdata['cluster'] == Bclu]        
            BreakfastItem = BreakfastItem.drop(['Breakfast','Lunch','Dinner','Snack','Veg','cluster'], axis=1)

        if(bmicl == 1):
            BreakfastItem = BreakfastfoodseparatedIDdata    
            BreakfastItem = BreakfastItem.drop(['Breakfast','Lunch','Dinner','Snack','Veg','cluster'], axis=1)
        
        for i in BreakfastItem.sample(frac=1).iterrows():
            if(BreakfastCaloriesCount + i[1]['Calories']) <= BreakfastCalories:

                if(len(BreakfastPackage) != 0):
                    foodName=BreakfastItem[BreakfastItem['Food_items'].isin(BreakfastPackage)]
                    randomItemType=i[1]['type']

                    #print(BreakfastPackageID)
                    if(randomItemType not in BreakfastPackageID):
                        BreakfastCaloriesCount += i[1]['Calories']
                        BreakfastProteinCount += i[1]['Protein']
                        BreakfastCarbsCount += i[1]['Carbs']
                        BreakfastFatCount += i[1]['Fat']
                        BreakfastFiberCount += i[1]['Fiber']
                        BreakfastPackage.append(i[1]['Food_items']) 
                        #print(i[1]['type'])
                        BreakfastPackageID.append(i[1]['type'])
            
                else:
                    BreakfastCaloriesCount += i[1]['Calories']
                    BreakfastProteinCount += i[1]['Protein']
                    BreakfastCarbsCount += i[1]['Carbs']
                    BreakfastFatCount += i[1]['Fat']
                    BreakfastFiberCount += i[1]['Fiber']
                    BreakfastPackage.append(i[1]['Food_items'])
                    BreakfastPackageID.append(i[1]['type'])
            
            if BreakfastCaloriesCount == BreakfastCalories:
                break

        BreakfastFoodPackage = BreakfastItem[BreakfastItem['Food_items'].isin(BreakfastPackage)]
        BreakfastFoodPackage.drop(['type','Calories','Protein','Carbs','Fat','Fiber'],axis=1,inplace=True);
        # print(Bclu)
        # print("Total Breakfast Calories Need:",BreakfastCalories)
        # print("Total Breakfast Calories Calculated:",BreakfastCaloriesCount)
        # print("Total Breakfast Proteins Calculated:",BreakfastProteinCount)
        # print("Total Breakfast Carbs Calculated:",BreakfastCarbsCount)
        # print("Total Breakfast Fat Calculated:",BreakfastFatCount)
        # print("Total Breakfast Fiber Calculated:",BreakfastFiberCount)

        # print("------------------------------------------------------------------------------")
        # print(BreakfastFoodPackage)


        ##################### Lunch Package ####################
        
        ### Here we will intialize Lunch Calories variable ###

        LunchPackage=[]

        LunchPackageID=[]

        LunchCaloriesCount = 0.0
        LunchProteinCount = 0.0
        LunchCarbsCount = 0.0
        LunchFatCount = 0.0
        LunchFiberCount = 0.0

        #------------user lunch calories
        # LunchCalories = 400 #---------------------------
        LunchCalories = int(LunchCalories)
        #-------------------

        Ltype = LunchfoodseparatedIDdata.iloc[LunchfoodseparatedIDdata['Calories'].argmin()]
        Lclu = Ltype['cluster']

        bmicl = 0

        if(bmicl == 0):
            LunchItem = LunchfoodseparatedIDdata.loc[LunchfoodseparatedIDdata['cluster'] == Lclu]  
            LunchItem = LunchItem.drop(['Breakfast','Lunch','Dinner','Snack','Veg','cluster'], axis=1)

        if(bmicl == 1):
            LunchItem = LunchfoodseparatedIDdata
            LunchItem = LunchItem.drop(['Breakfast','Lunch','Dinner','Snack','Veg','cluster'], axis=1)
        
        for i in LunchItem.sample(frac=1).iterrows():
            if(LunchCaloriesCount + i[1]['Calories']) <= LunchCalories:
                if(len(LunchPackage) != 0):
                    foodName=LunchItem[LunchItem['Food_items'].isin(LunchPackage)]
                    randomItemType=i[1]['type']
                    #print(LunchPackageID)
                    if(randomItemType not in LunchPackageID):
                            LunchCaloriesCount += i[1]['Calories']
                            LunchProteinCount += i[1]['Protein']
                            LunchCarbsCount += i[1]['Carbs']
                            LunchFatCount += i[1]['Fat']
                            LunchFiberCount += i[1]['Fiber']
                            LunchPackage.append(i[1]['Food_items'])
                            #print(i[1]['type'])
                            LunchPackageID.append(i[1]['type'])
        
                else:
                    LunchCaloriesCount += i[1]['Calories']
                    LunchProteinCount += i[1]['Protein']
                    LunchCarbsCount += i[1]['Carbs']
                    LunchFatCount += i[1]['Fat']
                    LunchFiberCount += i[1]['Fiber']
                    LunchPackage.append(i[1]['Food_items'])
                    LunchPackageID.append(i[1]['type'])
        
            if LunchCaloriesCount == LunchCalories:
                break

        LunchFoodPackage = LunchItem[LunchItem['Food_items'].isin(LunchPackage)] 
        LunchFoodPackage.drop(['type','Calories','Protein','Carbs','Fat','Fiber'],axis=1,inplace=True);

        # print(Lclu)
        # print("Total Lunch Calories Need:",LunchCalories)
        # print("Total Lunch Calories Calculated:",LunchCaloriesCount)
        # print("Total Lunch Proteins Calculated:",LunchProteinCount)
        # print("Total Lunch Carbs Calculated:",LunchCarbsCount)
        # print("Total Lunch Fat Calculated:",LunchFatCount)
        # print("Total Lunch Fiber Calculated:",LunchFiberCount)

        # print("------------------------------------------------------------------------------")
        # print(LunchFoodPackage)

        ##################### Snack Package ####################
        
        ### Here we will intialize Snack Calories variable ###

        SnackPackage=[]

        SnackPackageID=[]

        SnackCaloriesCount = 0.0
        SnackProteinCount = 0.0
        SnackCarbsCount = 0.0
        SnackFatCount = 0.0
        SnackFiberCount = 0.0

        #------------user snacks calories need
        # SnackCalories = 300      # ---------------------------------
        SnackCalories = int(SnackCalories)
        #-------------------

        Stype = SnackfoodseparatedIDdata.iloc[SnackfoodseparatedIDdata['Calories'].argmin()]
        Sclu = Stype['cluster']


        bmicl = 0

        if(bmicl == 0):
            SnackItem = SnackfoodseparatedIDdata.loc[SnackfoodseparatedIDdata['cluster'] == Sclu]  
            SnackItem = SnackItem.drop(['Breakfast','Lunch','Dinner','Snack','Veg','cluster'], axis=1)

        if(bmicl == 1):
            SnackItem = SnackfoodseparatedIDdata 
            SnackItem = SnackItem.drop(['Breakfast','Lunch','Dinner','Snack','Veg','cluster'], axis=1)
        
        for i in SnackItem.sample(frac=1).iterrows():
            if(SnackCaloriesCount + i[1]['Calories']) <= SnackCalories:
                if(len(SnackPackage) != 0):
                    foodName=SnackItem[SnackItem['Food_items'].isin(SnackPackage)]
                    randomItemType=i[1]['type']
                    #print(SnackPackageID)
                    if(randomItemType not in SnackPackageID):
                        SnackCaloriesCount += i[1]['Calories']
                        SnackProteinCount += i[1]['Protein']
                        SnackCarbsCount += i[1]['Carbs']
                        SnackFatCount += i[1]['Fat']
                        SnackFiberCount += i[1]['Fiber']
                        SnackPackage.append(i[1]['Food_items'])
                        #print(i[1]['type'])
                        SnackPackageID.append(i[1]['type'])
            
                else:
                    SnackCaloriesCount += i[1]['Calories']
                    SnackProteinCount += i[1]['Protein']
                    SnackCarbsCount += i[1]['Carbs']
                    SnackFatCount += i[1]['Fat']
                    SnackFiberCount += i[1]['Fiber']
                    SnackPackage.append(i[1]['Food_items'])
                    SnackPackageID.append(i[1]['type'])
            
            if SnackCaloriesCount == SnackCalories:
                break

        SnackFoodPackage = SnackItem[SnackItem['Food_items'].isin(SnackPackage)]
        SnackFoodPackage.drop(['type','Calories','Protein','Carbs','Fat','Fiber'],axis=1,inplace=True);

        #print(Sclu)
        # print("Total Snack Calories Need:",SnackCalories)
        # print("Total Snack Calories Calculated:",SnackCaloriesCount)
        # print("Total Snack Proteins Calculated:",SnackProteinCount)
        # print("Total Snack Carbs Calculated:",SnackCarbsCount)
        # print("Total Snack Fat Calculated:",SnackFatCount)
        # print("Total Snack Fiber Calculated:",SnackFiberCount)

        # print("------------------------------------------------------------------------------")
        # print(SnackFoodPackage)


        ##################### Dinner Package ####################
        
        ### Here we will intialize Dinner Calories variable ###

        DinnerPackage=[]

        DinnerPackageID=[]


        DinnerCaloriesCount = 0.0
        DinnerProteinCount = 0.0
        DinnerCarbsCount = 0.0
        DinnerFatCount = 0.0
        DinnerFiberCount = 0.0

        #------Dinner Calories need
        # DinnerCalories = 450            #------------------------
        DinnerCalories = int(DinnerCalories)
        #-------------------

        Dtype = DinnerfoodseparatedIDdata.iloc[DinnerfoodseparatedIDdata['Calories'].argmin()]
        Dclu = Dtype['cluster']

        bmicl = 0

        if(bmicl == 0):
            DinnerItem = DinnerfoodseparatedIDdata.loc[DinnerfoodseparatedIDdata['cluster'] == Dclu] 
            DinnerItem = DinnerItem.drop(['Breakfast','Lunch','Dinner','Snack','Veg','cluster'], axis=1)


        if(bmicl == 1):
            DinnerItem = DinnerfoodseparatedIDdata
            DinnerItem = DinnerItem.drop(['Breakfast','Lunch','Dinner','Snack','Veg','cluster'], axis=1)
        
        for i in DinnerItem.sample(frac=1).iterrows():
            if(DinnerCaloriesCount + i[1]['Calories']) <= DinnerCalories:
                if(len(DinnerPackage) != 0):
                    foodName=DinnerItem[DinnerItem['Food_items'].isin(DinnerPackage)]
                    randomItemType=i[1]['type']
                    #print(DinnerPackageID)
                    if(randomItemType not in DinnerPackageID):
                        DinnerCaloriesCount += i[1]['Calories']
                        DinnerProteinCount += i[1]['Protein']
                        DinnerCarbsCount += i[1]['Carbs']
                        DinnerFatCount += i[1]['Fat']
                        DinnerFiberCount += i[1]['Fiber']
                        DinnerPackage.append(i[1]['Food_items'])
                        #print(i[1]['type'])
                        DinnerPackageID.append(i[1]['type'])
            
                else:
                    DinnerCaloriesCount += i[1]['Calories']
                    DinnerProteinCount += i[1]['Protein']
                    DinnerCarbsCount += i[1]['Carbs']
                    DinnerFatCount += i[1]['Fat']
                    DinnerFiberCount += i[1]['Fiber']
                    DinnerPackage.append(i[1]['Food_items'])
                    DinnerPackageID.append(i[1]['type'])
            
            if DinnerCaloriesCount == DinnerCalories:
                break

        DinnerFoodPackage =  DinnerItem[DinnerItem['Food_items'].isin(DinnerPackage)] 
        DinnerFoodPackage.drop(['type','Calories','Protein','Carbs','Fat','Fiber'],axis=1,inplace=True);

        # print(Dclu)
        # print("Total Dinner Calories Need:",DinnerCalories)
        # print("Total Dinner Calories Calculated:",DinnerCaloriesCount)
        # print("Total Dinner Proteins Calculated:",DinnerProteinCount)
        # print("Total Dinner Carbs Calculated:",DinnerCarbsCount)
        # print("Total Dinner Fat Calculated:",DinnerFatCount)
        # print("Total Dinner Fiber Calculated:",DinnerFiberCount)

        # print("------------------------------------------------------------------------------")
        # print(DinnerFoodPackage)

        # TotalCalories = BreakfastCaloriesCount + LunchCaloriesCount + SnackCaloriesCount + DinnerCaloriesCount
        # TotalProtein = BreakfastProteinCount + LunchProteinCount + SnackProteinCount + DinnerProteinCount
        # TotalCarbs = BreakfastCarbsCount + LunchCarbsCount + SnackCarbsCount + DinnerCarbsCount
        # TotalFat = BreakfastFatCount + LunchFatCount + SnackFatCount + DinnerFatCount
        # TotalFiber = BreakfastFiberCount + LunchFiberCount + SnackFiberCount + DinnerFiberCount



        # print("----------------------My Diet Plan-----------------------------------------\n")

        # print("BMICL", bmicl,"\n")
        # print("**********************Breakfast Plan***************************************\n")
        # print("Total Breakfast Calories Need:",BreakfastCalories)
        # print("Total Breakfast Calories Calculated:",BreakfastCaloriesCount,"\n")
        # print(BreakfastFoodPackage,"\n")
        # print("**********************Lunch Plan***************************************\n")
        # print("Total Lunch Calories Need:",LunchCalories)
        # print("Total Lunch Calories Calculated:",LunchCaloriesCount,"\n")
        # print(LunchFoodPackage,"\n")
        # print("**********************Snack Plan***************************************\n")
        # print("Total Snack Calories Need:",SnackCalories)
        # print("Total Snack Calories Calculated:",SnackCaloriesCount,"\n")
        # print(SnackFoodPackage,"\n")
        # print("**********************Dinner Plan***************************************\n")
        # print("Total Dinner Calories Need:",DinnerCalories)
        # print("Total Dinner Calories Calculated:",DinnerCaloriesCount,"\n")
        # print(DinnerFoodPackage,"\n")

        res= make_response({"message": "Diet Plan generated successfully",
                            "status" : "SUCCESS",
                            "code" : 200,
                            "payload":{
                                "BreakfastFoodPackage": format(BreakfastFoodPackage),
                                "LunchFoodPackage": format(LunchFoodPackage),
                                "SnackFoodPackage": format(SnackFoodPackage),
                                "DinnerFoodPackage": format(DinnerFoodPackage),
                                # "BreakfastCaloriesCount": format(BreakfastCaloriesCount),
                                # "LunchCaloriesCount": format(LunchCaloriesCount),
                                # "SnackCaloriesCount": format(SnackCaloriesCount),
                                # "DinnerCaloriesCount": format(DinnerCaloriesCount),
                                # "TotalCaloriesConsumed": format(TotalCalories),
                                # "TotalProteinConsumed": format(TotalProtein),
                                # "TotalCarbsConsumed": format(TotalCarbs),
                                # "TotalFatConsumed": format(TotalFat),
                                # "TotalFiberConsumed": format(TotalFiber),
                                
                            }}
                            , 200)
        res.headers['Access-Control-Allow-Origin'] = "*"          
        return res

################################# Route ######################################
obj = user_dietPlan_model()

@app.route("/user_dietPlan", methods=["POST"])
def user_dietPlan_controller():
    return obj.user_dietPlan_method(request.args)

plt.switch_backend('agg')

###### Testing #####
    
@app.route("/test")
def test():
    return "Deployed"
