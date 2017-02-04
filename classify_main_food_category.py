
#Import Dependencies
import pandas as pd
import numpy as np
import sklearn
import re
from pandas_ml import ConfusionMatrix
import model

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from collections import defaultdict


def process_quantity_column(text):
    try:
        characters = list(text)
        if 'g' in characters or 'o' in characters:
            return 'solid'
        elif 'l' in characters:
            return 'liquid'
        else:
            return 'unknown'
        
    except:
        return 'unknown'
    
def extract_nums_from_quanity(text):
    try:
        nums = re.findall(r'\d+', text)
        return float(nums[0])
    except:
        return 0.0
    
if __name__ == '__main__':
    
    #Data Processing
    food_facts = pd.read_csv('ffclean6.csv')
    food_impute = food_facts.fillna(food_facts.mean())

    #impute with column mode
    food_impute_mode = food_facts.apply(lambda x:x.fillna(x.value_counts().index[0]))
    
    food_facts['food_type'] = food_facts['quantity'].apply(lambda x: process_quantity_column(x))

    food_facts['quantity_num'] = food_facts['quantity'].apply(lambda x: extract_nums_from_quanity(x))

    food_facts_clean_quantities = food_facts.drop('quantity', 1)
    
    #Predict Main Categories
    vcs = food_facts_clean_quantities.main_category_en.value_counts()[:20]
    
    food_cat_subset = food_facts_clean_quantities.loc[food_facts_clean_quantities['main_category_en'].isin(vcs)]
    
    food_df = pd.concat([food_cat_subset, pd.get_dummies(food_cat_subset.main_category_en)], axis=1)

    food_df = food_df[['additives_n', 'ingredients_that_may_be_from_palm_oil_n', 'energy_100g', 'fat_100g', 'saturated_fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g', 'nutrition_score_fr_100g', 'quantity_num']+vcs]
    food_df = food_df.dropna()

    #Train Test Split
    food_X = food_df.drop(vcs, 1)
    food_y = food_df[vcs]

    x_train, x_test, y_train, y_test = train_test_split(food_X, food_y, test_size=0.3, random_state=42)
    
    #Train Random Forest Model
    rf_model.fit(x_train, y_train)
​
    #Generate predtions on train and test splits
    predictions_train = rf_model.predict(x_train)
    predictions_validate = rf_model.predict(x_test)
    ​
    #Calculate accuracy scores and store in container
    accuracy = accuracy_score(y_train, predictions_train)
    accuracy_validate = accuracy_score(y_test, predictions_validate)
    ​
    ​
    #Calculate precision scores and store in container
    precision_train = precision_score(y_train, predictions_train, average='macro')
    precision_validate = precision_score(y_test, predictions_validate, average='macro')
    ​
    ​
    #Calculate recall scores and store in container
    recall_train = recall_score(y_train, predictions_train, average='macro')
    recall_validate = recall_score(y_test, predictions_validate, average='macro')
    ​
    #Calculate f1 scores and store in container
    f1_train = f1_score(y_train, predictions_train, average='macro')
    f1_validate = f1_score(y_test, predictions_validate, average='macro')
    
    rf_results = {'accuracy': accuracy_validate, 'precision':precision_validate ,'recall': recall_validate,'f1':f1_validate}
    
    print(rf_results)
    
