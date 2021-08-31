#import packages
import sys
import math
import numpy as np
import udacourse2 #my library for this project!
import pandas as pd
from time import time

#SQLAlchemy toolkit
from sqlalchemy import create_engine
from sqlalchemy import pool
from sqlalchemy import inspect

#Machine Learning preparing/preprocessing toolkits
from sklearn.model_selection import train_test_split

#Machine Learning Feature Extraction tools
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#Machine Learning Classifiers
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

#Machine Learning Classifiers extra tools
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

#pickling tool
import pickle

#only a dummy function, as I pre-tokenize my data
def dummy(doc):
    return doc

#########1#########2#########3#########4#########5#########6#########7#########8
def load_data(data_file, 
              verbose=False):
    '''This function takes a path for a MySQL table and returns processed data
    for training a Machine Learning Classifier
    Inputs:
      - data_file (mandatory) - full path for SQLite table - text string
      - verbose (optional) - if you want some verbosity during the running 
        (default=False)
    Outputs:
      - X - tokenized text X-training - Pandas Series
      - y - y-multilabels 0|1 - Pandas Dataframe'''
    if verbose:
        print('###load_data function started')
    start = time()

    #1.read in file
    #importing MySQL to Pandas - load data from database
    engine = create_engine(data_file, poolclass=pool.NullPool) #, echo=True)
    #retrieving tables names from my DB
    inspector = inspect(engine)
    if verbose:
        print('existing tables in my SQLite database:', inspector.get_table_names())
    connection = engine.connect()
    df = pd.read_sql('SELECT * FROM Messages', con=connection)
    connection.close()
    df.name = 'df'
    
    #2.clean data
    #2.1.Elliminate rows with all-blank labels
    if verbose:
        print('all labels are blank in {} rows'.format(df[df['if_blank'] == 1].shape[0]))
    df = df[df['if_blank'] == 0]
    if verbose:
        print('remaining rows:', df.shape[0])
    #Verifying if removal was complete
    if df[df['if_blank'] == 1].shape[0] == 0:
        if verbose:
            print('removal complete!')
    else:
        raise Exception('something went wrong with {} rows to remove'.\
format(df[df['if_blank'] == 1].shape[0]))
            
    #2.2.Premature Tokenization Strategy (pre-tokenizer)
    #Pre-Tokenizer + not removing provisory tokenized column
    #inserting a tokenized column
    try:
        df = df.drop('tokenized', axis=1)
    except KeyError:
        if verbose:
            print('OK')
    df.insert(1, 'tokenized', np.nan)

    #tokenizing over the provisory
    df['tokenized'] = df.apply(lambda x: udacourse2.fn_tokenize_fast(x['message']), axis=1)

    #removing NaN over provisory (if istill exist)
    df = df[df['tokenized'].notnull()]
    empty_tokens = df[df['tokenized'].apply(lambda x: len(x)) == 0].shape[0]
    if verbose:
        print('found {} rows with no tokens'.format(empty_tokens))
    df = df[df['tokenized'].apply(lambda x: len(x)) > 0]
    empty_tokens = df[df['tokenized'].apply(lambda x: len(x)) == 0].shape[0]
    if verbose:
        print('*after removal, found {} rows with no tokens'.format(empty_tokens))

    #I will drop the original 'message' column
    try:
        df = df.drop('message', axis=1)
    except KeyError:
        if verbose:
            print('OK')
    if verbose:
        print('now I have {} rows to train'.format(df.shape[0]))

    #2.3.Database Data Consistency Check/Fix
    #correction for aid_related
    df = udacourse2.fn_group_check(dataset=df,
                                   subset='aid',
                                   correct=True, 
                                   shrink=False, 
                                   shorten=False, 
                                   verbose=verbose)
    #correction for weather_related
    df = udacourse2.fn_group_check(dataset=df,
                                   subset='wtr',
                                   correct=True, 
                                   shrink=False, 
                                   shorten=False, 
                                   verbose=verbose)
    #correction for infrastrucutre_related
    df = udacourse2.fn_group_check(dataset=df,
                                   subset='ifr',
                                   correct=True, 
                                   shrink=False, 
                                   shorten=False, 
                                   verbose=verbose)
    #correction for related(considering that the earlier were already corrected)
    df = udacourse2.fn_group_check(dataset=df,
                                   subset='main',
                                   correct=True, 
                                   shrink=False, 
                                   shorten=False, 
                                   verbose=verbose)
    
    #load to database <-I don't know for what it is
    
    #3.Define features and label arrays (break the data)
    #3.1.X is the Training Text Column
    X = df['tokenized']
    
    #3.2.y is the Classification labels
    #I REMOVED "related" column from my labels, as it is impossible to train it!
    y = df[df.columns[5:]]
    remove_lst = []

    for column in y.columns:
        col = y[column]
        if (col == 0).all():
            if verbose:
                print('*{} -> only zeroes training column!'.format(column))
            remove_lst.append(column)
        else:
            #print('*{} -> column OK'.format(column))
            pass
        
    if verbose:
        print(remove_lst)
    y = y.drop(remove_lst, axis=1)
    
    spent = time() - start
    if verbose:
        print('*dataset breaked into X-Training Text Column and Y-Multilabels')    
        print('process time:{:.0f} seconds'.format(spent))
    return X, y

#########1#########2#########3#########4#########5#########6#########7#########8
def build_model(verbose=False):
    '''This function builds the Classifier Pipeline, for future fitting
    Inputs:
      - verbose (optional) - if you want some verbosity during the running 
        (default=False)
    Output:
      - model_pipeline for your Classifiear (untrained)
    '''
    if verbose:
        print('###build_model function started')
    start = time()
    
    #1.text processing and model pipeline
    #(text processing was made at a earlier step, at Load Data function)
    feats = TfidfVectorizer(analyzer='word', 
                            tokenizer=dummy, 
                            preprocessor=dummy,
                            token_pattern=None,
                            ngram_range=(1, 3))
    
    classif = OneVsRestClassifier(LinearSVC(C=2., 
                                            random_state=42))
    
    model_pipeline = Pipeline([('vect', feats),
                               ('clf', classif)])
    
    #define parameters for GridSearchCV (parameters already defined)
    #create gridsearch object and return as final model pipeline (made at pipeline preparation)
    #obs: for better performance, I pre-tokenized my data. And GridSearch was runned on Jupyter,
    #     and the best parameters where adjusted, just to save processing time during code execution.
    spent = time() - start
    if verbose:
        print('*Linear Support Vector Machine pipeline was created')
        print('process time:{:.0f} seconds'.format(spent))
    return model_pipeline

#########1#########2#########3#########4#########5#########6#########7#########8
def train(X, 
          y, 
          model, 
          verbose=False):
    '''This function trains your already created Classifier Pipeline
    Inputs:
      - X (mandatory) - tokenized data for training - Pandas Series
      - y (mandatory) - Multilabels 0|1 - Pandas Dataset
      - verbose (optional) - if you want some verbosity during the running 
        (default=False)
    Output:
      - trained model'''
    if verbose:
        print('###train function started')
    start = time()

    #1.Train test split
    #Split makes randomization, so random_state parameter was set
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.25, 
                                                        random_state=42)
    if (X_train.shape[0] + X_test.shape[0]) == X.shape[0]:
        if verbose:
            print('data split into train and text seems OK')
    else:
        raise Exception('something went wrong when splitting the data')
        
    #2.fit the model
    model.fit(X_train, y_train)
    
    # output model test results
    y_pred = model.predict(X_test)
    if verbose:
        metrics = udacourse2.fn_scores_report2(y_test, 
                                               y_pred,
                                               best_10=True,
                                               verbose=True)
    else:
        metrics = udacourse2.fn_scores_report2(y_test, 
                                               y_pred,
                                               best_10=True,
                                               verbose=False)

    for metric in metrics:
        if metric < 0.6:
            raise Exception('something is wrong, model is predicting poorly')

    spent = time() - start
    if verbose:
        print('*classifier was trained!')
        print('process time:{:.0f} seconds'.format(spent))
    return model

#########1#########2#########3#########4#########5#########6#########7#########8
def export_model(model,
                 file_name='classifier.pkl',
                 verbose=False):
    '''This function writes your already trained Classifiear as a Picke Binary
    file.
    Inputs:
      - model (mandatory) - your already trained Classifiear - Python Object
      - file_name (optional) - the name of the file to be created (default:
         'classifier.pkl')
      - verbose (optional) - if you want some verbosity during the running 
        (default=False)
       Output: return True if everything runs OK
      ''' 
    if verbose:
        print('###export_model function started')
    start = time()

    #1.Export model as a pickle file
    file_name = file_name

    #writing the file
    with open (file_name, 'wb') as pk_writer: 
        pickle.dump(model, pk_writer)

    #reading the file
    #with open('classifier.pkl', 'rb') as pk_reader:
    #    model = pickle.load(pk_reader)
    
    spent = time() - start
    if verbose:
        print('*trained Classifier was exported')
        print('process time:{:.0f} seconds'.format(spent))
        
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def run_pipeline(data_file='sqlite:///Messages.db', 
                 verbose=False):
    '''This function is a caller: it calls load, build, train and save modules
    Inputs:
      - data_file (optional) - complete path to the SQLite datafile to be 
        processed - (default='sqlite:///Messages.db')
      - verbose (optional) - if you want some verbosity during the running 
        (default=False)
    Output: return True if everything runs OK
    '''
    if verbose:
        print('###run_pipeline function started')
    start = time()

    #1.Run ETL pipeline
    X, y = load_data(data_file, 
                     verbose=verbose)
    #2.Build model pipeline
    model = build_model(verbose=verbose)
    #3.Train model pipeline
    model = train(X, 
                  y, 
                  model, 
                  verbose=verbose)
    # save the model
    export_model(model,
                 verbose=verbose)
    
    spent = time() - start
    if verbose:
        print('process time:{:.0f} seconds'.format(spent))
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def main(verbose=False):
    '''This is the main Machine Learning Pipeline function. It calls the other 
    ones, in the correct order.
    Imputs:
      - verbose (optional) - if you want some verbosity during the running 
        (default=False)
    '''
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file='sqlite:///Messages.db',
                 verbose=verbose)

#########1#########2#########3#########4#########5#########6#########7#########8
if __name__ == '__main__':
    main()
