import numpy as np
import pandas as pd
import os

def read_data():
    # set path to raw data
    raw_data_path = os.path.join(os.path.pardir,'data','raw')
    train_file_path = os.path.join(raw_data_path,'train.csv')
    test_file_path = os.path.join(raw_data_path,'test.csv')
    # read data with default parameters
    train_df = pd.read_csv(train_file_path,index_col='PassengerId')
    test_df = pd.read_csv(test_file_path,index_col='PassengerId')
    test_df['Survived'] = -888 # Survived in test data
    df = pd.concat((train_df,test_df),axis=0)
    return df

def process_data(df):
    return (df
           .assign(Title = lambda x : x.Name.map(getTitle)) # x indicates a complete dataframe
           .pipe(fill_missing_values)
           .assign(Fare_category = lambda x : pd.qcut(x.Fare,4,labels=['very_low','low','high','very_high']))
           .assign(AgeState = lambda x : np.where(x.Age>=18,'Adult','Child'))
           .assign(FamilySize = lambda x : x.Parch + x.SibSp + 1)
           .assign(IsMother = lambda x : np.where(((x.Sex == 'female') & (x.Parch>0) & (x.Age>18) & (x.Title != 'Miss')),1,0))
           .assign(Cabin = lambda x : np.where(x.Cabin == 'T',np.nan,x.Cabin))
           .assign(Deck = lambda x : x.Cabin.map(get_deck))
           .assign(IsMale = lambda x : np.where(x.Sex == 'male',1,0))
           .pipe(pd.get_dummies,columns = ['Deck','Pclass','Title','Fare_category','Embarked','AgeState'])
           .drop(['Cabin','Name','Parch','Sex','SibSp','Ticket'],axis=1)
           .pipe(reorder_columns)
        )

def getTitle(name):
    title_group = {
        'mr' : 'Mr',
        'mrs' : 'Mrs',
        'miss': 'Miss',
        'master' : 'Master',
        'don' : 'Sir',
        'rev' : 'Sir',
        'dr' : 'Officer',
        'mme' : 'Mrs',
        'ms' : 'Mrs',
        'major' : 'Officer',
        'lady' : 'Lady',
        'sir' : 'Sir',
        'mlle' : 'Miss',
        'col' : 'Officer',
        'capt': 'Officer',
        'the countess' : 'Lady',
        'jonkheer' : 'Sir',
        'dona' : 'Lady'
    }
    first_name = name.split(',')[1] # split wrt comma and extract the string from Mr or Mrs
    title = first_name.split('.')[0] # obtain Mr or Mrs
    title = title.strip().lower() # strip out white spaces
    return title_group[title]

def fill_missing_values(df):
    df.Embarked.fillna('C',inplace=True)
    median_fare = df.loc[(df.Pclass==3) & (df.Embarked=='S'),'Fare'].median()
    df.Fare.fillna(median_fare,inplace=True)
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median,inplace=True)
    return df
    
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'z')

def reorder_columns(df):
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    df = df[columns]
    return df

def write_data(df):
    processed_data_path = os.path.join(os.path.pardir,'data','processed')
    write_train_path = os.path.join(processed_data_path,'train.csv')
    write_test_path = os.path.join(processed_data_path,'test.csv')
    df.loc[df.Survived != -888].to_csv(write_train_path) # train data
    columns = [column for column in df.columns if column != 'Survived'] # test data
    df.loc[df.Survived == -888,columns].to_csv(write_test_path)

if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)