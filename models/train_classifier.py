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
from sklearn.model_selection import GridSearchCV

#Machine Learning Feature Extraction tools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#Machine Learning Classifiers
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier

#Machine Learning Classifiers extra tools
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

#pickling tool
import pickle

#only a dummy function, as I pre-tokenize my data
def dummy(doc):
    return doc

#########1#########2#########3#########4#########5#########6#########7#########8
def load_data(data_file, #='sqlite:///Messages.db',
              remove_cols=False,
              pre_tokenize=False,
              condense=True,
              verbose=False):
    '''This function takes a path for a MySQL table and returns processed data
    for training a Machine Learning Classifier
    Inputs:
      - data_file (mandatory) - full path for SQLite table - text string
      - remove_cols (optional) - if you want to remove (un)trainable labels
        columns (default=False)
      - pre_tokenize (optional) - if you want to keep preprocessing tokenization
        column, thaw was made for removal of empty-messages from X-training.
        Obsservation: keeping this column turns the system faster, but it can
        cause instability on Classifier training on Flask due to "pipeline
        leakage" (default=False)
      - condense (optional) - if you want to remove duplicate tokens under
        one document (default=True)
        See: https://rebeccabilbro.github.io/module-main-has-no-attribute/
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
            print('*tokenized column don´t exist, creating it')
    df.insert(1, 'tokenized', np.nan)

    #tokenizing over the provisory
    df['tokenized'] = df.apply(lambda x: udacourse2.fn_tokenize_fast(x['message']), axis=1) #condense=condense
    
    #removing NaN over provisory (if istill exist)
    df = df[df['tokenized'].notnull()]
    empty_tokens = df[df['tokenized'].apply(lambda x: len(x)) == 0].shape[0]
    if verbose:
        print('found {} rows with no tokens'.format(empty_tokens))
    df = df[df['tokenized'].apply(lambda x: len(x)) > 0]
    empty_tokens = df[df['tokenized'].apply(lambda x: len(x)) == 0].shape[0]
    if verbose:
        print('*after removal, found {} rows with no tokens'.format(empty_tokens))

    if pre_tokenize:
        #I will drop the original 'message' column
        try:
            df = df.drop('message', axis=1)
            if verbose:
                print('*message column dropped')
        except KeyError:
            if verbose:
                print('*message column not dropped')
    else:
        #I will drop the already processed 'tokenized' column        
        try:
            df = df.drop('tokenized', axis=1)
            if verbose:
                print('*tokenized column dropped')
        except KeyError:
            if verbose:
                print('*tokenized column not dropped')
            
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
    
    #load to database <-I don't know for what it is!
    
    #3.Define features and label arrays (break the data)
    #3.1.X is the Training Text Column
    if pre_tokenize:
        if verbose:
            print('X-training will be the tokenized column')
        X = df['tokenized']
    else:
        if verbose:
            print('X-training will be the message column')
        X = df['message']
    
    #3.2.y is the Classification labels
    #I REMOVED "related" column from my labels, as it is impossible to train it!
    if remove_cols: #for removing untrainable columns
        y = df[df.columns[5:]] #for removal of "related" column
        remove_lst = []

        for column in y.columns:
            col = y[column]
            if (col == 0).all():
                if verbose:
                    print('*{} -> only zeroes (un)training column!'.format(column))
                remove_lst.append(column)
            elif (col == 1).all():
                if verbose:
                    print('*{} -> only ones (un)training column!'.format(column))
                remove_lst.append(column)
            else:
                if verbose:
                    print('*{} -> column OK'.format(column))
                pass
        if verbose:
            print('remove colums:', remove_lst)
        y = y.drop(remove_lst, axis=1)
        if verbose:
            print('(un)trainable label columns removed')
    else:
        y = df[df.columns[4:]]
        if y.shape[1] == 36:
            if verbose:
                print('y dataset has 36 labels')
        else:
            raise Exception('something went wrong, dataset has {} labels instead of 36'.format(y.shape[1]))
    
    spent = time() - start
    if verbose:
        print('*dataset breaked into X-Training Text Column and Y-Multilabels')    
        print('process time:{:.0f} seconds'.format(spent))
    return X, y

#########1#########2#########3#########4#########5#########6#########7#########8
def build_model(data_file, #='sqlite:///Messages.db',
                tree_type=True,
                C=2.0,
                learning_rate=0.5,
                n_estimators=80,
                remove_cols=False,
                grid_search=False,
                pre_tokenize=False,
                condense=True,
                test_size=0.25,
                verbose=False):
    '''This function builds the Classifier Pipeline, for future fitting
    Inputs:
      - data_file (mandatory) - full path for the source SQLite file
      - tree-type (optional) - Classifier will be from Tree-family (Adaboost)
        if you want to use the alternative Classifier (LSVM), set it as False.
        Default: True
      - C (optional) - C parameter for the LinearSVC Classifier (default=2.0)
      - learning_rate (optional) - parameter for the Adaboost Classifier
        (default=0.5)
      - n_estimators (optional) - parameter for the Adaboost Classifier
        (default=80)      
      - remove_cols: ONLY used for Grid Search! (default=False)
      - grid_search (optional) performas Grid Search over Adaboost before 
        training for the best parameters. Please use it wisely, as it costs 
        a lot of processing time! (default=False)
        Observation: Grid Search was not implemented on this code for the 
        alternative Classifier (LSVM), so this parameter will not work when 
        using (tree_type=False)
      - pre_tokenize (optional) - if you want to keep preprocessing tokenization
        column, thaw was made for removal of empty-messages from X-training.
        Obsservation: keeping this column turns the system faster, but it can
        cause instability on Classifier training on Flask due to "pipeline
        leakage" (default=False)
        See: https://rebeccabilbro.github.io/module-main-has-no-attribute/
      - condense (optional) - if you want to remove duplicate tokens under
        one document (default=True) note: not activated due to technical issues
      - test_size: ONLY used for Grid Sarch! (default=0.25)
      - verbose (optional) - if you want some verbosity during the running 
        (default=False)
    Output:
      - model_pipeline for your Classifiear (untrained)
    Future improvement:
      - other parameters for Grid Search
      - Grid Search for the alternative Classifier
    '''
    if verbose:
        print('###build_model function started')
    start = time()
    
    #0.grid search for Adaboost Classifier
    if grid_search:
        if verbose:
            print('###Grid Search function started over Adaboost')
            print('WARNING: running parameters optimization requires hard processing and some repetitive processes!')
        vect_a = CountVectorizer(tokenizer=udacourse2.fn_tokenize_fast)
        transf_a = TfidfTransformer()
        class_a = MultiOutputClassifier(AdaBoostClassifier(random_state=42))

        pipeline_adab = Pipeline([('vect', vect_a),
                                  ('tfidf', transf_a),
                                  ('clf', class_a)])

        param_dict = {'clf__estimator__learning_rate': [0.5, 1.0],
                      'clf__estimator__n_estimators': [50, 80]}

        estimator = GridSearchCV(estimator=pipeline_adab, 
                                 param_grid=param_dict,
                                 n_jobs=-1) #set for using maximum CPU cores

        X, y = load_data(data_file=data_file,
                         remove_cols=remove_cols,
                         pre_tokenize=pre_tokenize,
                         condense=condense,
                         verbose=verbose)

        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=test_size, 
                                                            random_state=42)
        pipeline_adab.fit(X_train, y_train)
        estimator_fit = train(X=X_train, 
                              y=y_train, 
                              model=estimator,
                              verbose=verbose)
        #get the best parameters
        learning_rate = estimator.best_estimator_[2].estimator.learning_rate
        n_estimators = estimator.best_estimator_[2].estimator.n_estimators
        if verbose:
            print('Grid Search finished - new params: learning_rate={}, n_estimators={}'\
                  .format(learning_rate, n_estimators))
    
    #1.text processing and model pipeline
    #(text processing was made at a earlier step, at Load Data function)   
    if tree_type:
        if verbose:
            print('Tree-type Classifier (Adaboost-default) pipeline is on the way')
            print('*note: parameter C, is NOT used in this family of Classifiers, so don´t call it!')
            
        if pre_tokenize:
            if verbose:
                print('creating pre-tokenized Adaboost pipeline')
                
            vec_adab = CountVectorizer(tokenizer=dummy, preprocessor=dummy)
            trf_adab = TfidfTransformer()
            clf_adab = MultiOutputClassifier(AdaBoostClassifier(learning_rate=learning_rate,
                                                                n_estimators=n_estimators,
                                                                random_state=42))
            model_pipeline = Pipeline([('vect', vec_adab),
                                       ('tfidf', trf_adab),
                                       ('clf', clf_adab)])
        else:
            if verbose:
                print('creating convencional Adaboost pipeline')
            vec_adab = CountVectorizer(tokenizer=udacourse2.fn_tokenize_fast)
            trf_adab = TfidfTransformer()
            clf_adab = MultiOutputClassifier(AdaBoostClassifier(learning_rate=learning_rate,
                                                                n_estimators=n_estimators,
                                                                random_state=42))
            model_pipeline = Pipeline([('vect', vec_adab),
                                       ('tfidf', trf_adab),
                                       ('clf', clf_adab)])
          
    else: #alternative LSVM Classifier
        if verbose:
            print('Support Vector Machine (Linear-alternative) pipeline is on the way')
            print('*note: parameters learning_rate and n_estimators are NOT used in this family of Classifiers')
            
        if pre_tokenize:
            if verbose:
                print('creating pre-tokenized LSVM pipeline')
                
            vec_lsvm = TfidfVectorizer(analyzer='word', 
                                       tokenizer=dummy, 
                                       preprocessor=dummy,
                                       token_pattern=None,
                                       ngram_range=(1, 3))
    
            cls_lsvm = OneVsRestClassifier(LinearSVC(C=C, 
                                                     random_state=42))
    
            model_pipeline = Pipeline([('vect', vec_lsvm),
                                       ('clf', cls_lsvm)])
        else:      
            if verbose:
                print('creating convencional LSVM pipeline')
                
            vec_lsvm = TfidfVectorizer(analyzer='word', 
                                       tokenizer=udacourse2.fn_tokenize_fast, 
                                       token_pattern=None,
                                       ngram_range=(1, 3))
    
            cls_lsvm = OneVsRestClassifier(LinearSVC(C=C, 
                                                     random_state=42))
    
            model_pipeline = Pipeline([('vect', vec_lsvm),
                                       ('clf', cls_lsvm)])
    
    #define parameters for GridSearchCV (parameters already defined)
    #create gridsearch object and return as final model pipeline (made at pipeline preparation)
    #obs: for better performance, I pre-tokenized my data. And GridSearch was runned on Jupyter,
    #     and the best parameters for LSVM where adjusted, just to save processing time during code execution.
    spent = time() - start
    if verbose:
        print('*Classifier pipeline was created')
        print('process time:{:.0f} seconds'.format(spent))
    return model_pipeline

#########1#########2#########3#########4#########5#########6#########7#########8
def train(X, 
          y, 
          model,
          test_size=0.25,
          best_10=True,
          verbose=False):
    '''This function trains your already created Classifier Pipeline
    Inputs:
      - X (mandatory) - tokenized data for training - Pandas Series
      - y (mandatory) - Multilabels 0|1 - Pandas Dataset
      - test_size (optional) - test size for data split (default=0.25)
      - best_10 (optional) - if metrics will be best_10 or all 
        (default=True - best_10)        
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
                                                        test_size=test_size, 
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
                                               best_10=best_10,
                                               data_ret=True,
                                               verbose=True)
    else:
        metrics = udacourse2.fn_scores_report2(y_test, 
                                               y_pred,
                                               best_10=best_10,
                                               data_ret=True,
                                               verbose=False)
    for metric in metrics:
        if metric < 0.6:
            if verbose:
                print('*metrics alert: something is wrong, model is predicting poorly')

    spent = time() - start
    if verbose:
        print('*classifier was trained!')
        print('process time:{:.0f} seconds'.format(spent))
    return model

#########1#########2#########3#########4#########5#########6#########7#########8
def export_model(model,
                 file_name, #='classifier.pkl',
                 verbose=False):
    '''This function writes your already trained Classifiear as a Picke Binary
    file.
    Inputs:
      - model (mandatory) - your already trained Classifiear - Python Object
      - file_name (mandatory) - the name of the file to be created
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
def run_pipeline(data_file, #='sqlite:///Messages.db',
                 file_name, #='classifier.pkl',
                 remove_cols=False,
                 tree_type=True,
                 C=2.0,
                 learning_rate=0.5,
                 n_estimators=80,
                 test_size=0.25,
                 best_10=True,
                 pre_tokenize=False,
                 condense=True,
                 grid_search=False,
                 verbose=False):
    '''This function is a caller: it calls load, build, train and save modules
    Inputs:
      - data_file (mandatory) - complete path to the SQLite datafile to be 
        processed
      - file_name (mandatory) - name for pickling the Classifier
      - remove_cols (optional) - if you want to remove (un)trainable labels
        columns (default=False)
      - tree-type (optional) - Classifier will be from Tree-family (Adaboost)
        if you want to use the alternative Classifier (LSVM), set it as False.
        Default: True
      - C (optional) - parameter for the LinearSVC Classifier (default=2.0)
      - learning_rate (optional) - parameter for the Adaboost Classifier
        (default=0.5)
      - n_estimators (optional) - parameter for the Adaboost Classifier
        (default=80)
      - test_size (optional) - test size for data split (default=0.25)
      - best_10 (optional) - if metrics will be best_10 or all 
        (default=True - best_10)        
      - pre_tokenize (optional) - if you want to keep preprocessing tokenization
        column, thaw was made for removal of empty-messages from X-training.
        Obsservation: keeping this column turns the system faster, but it can
        cause instability on Classifier training on Flask due to "pipeline
        leakage" (default=False)
      - condense (optional) - if you want to remove duplicate tokens under
        one document (default=True)
        See: https://rebeccabilbro.github.io/module-main-has-no-attribute/
      - grid_search (optional) performas Grid Search over Adaboost before 
        training for the best parameters. Please use it wisely, as it costs 
        a lot of processing time! (default=False).
        Observation: Grid Search was not implemented on this code for the 
        alternative Classifier (LSVM), so this parameter will not work when 
        using tree_type=False
      - verbose (optional) - if you want some verbosity during the running 
        (default=False)
    Output: return True if everything runs OK
    '''
    if verbose:
        print('###run_pipeline function started')
    start = time()

    #1.Run ETL pipeline
    X, y = load_data(data_file,
                     remove_cols=remove_cols,
                     pre_tokenize=pre_tokenize,
                     condense=condense,
                     verbose=verbose)
                                       
    #2.Build model pipeline
    if grid_search: #takes additional parameters for run
        model = build_model(C=C,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            tree_type=tree_type,
                            pre_tokenize=pre_tokenize,
                            remove_cols=remove_cols,
                            condense=condense,
                            grid_search=True,
                            test_size=test_size,
                            data_file=data_file,
                            verbose=verbose)
                                       
    else: #takes standard parameters only
        model = build_model(C=C,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            tree_type=tree_type,
                            pre_tokenize=pre_tokenize,
                            remove_cols=remove_cols,
                            condense=condense,
                            test_size=test_size,
                            grid_search=False,
                            data_file=data_file,
                            verbose=verbose)

    #3.Train model pipeline
    model = train(X, 
                  y, 
                  model,
                  test_size=test_size,
                  best_10=best_10,
                  verbose=verbose)
                                                                              
    # save the model
    export_model(model,
                 file_name=file_name,
                 verbose=verbose)
    
    spent = time() - start
    if verbose:
        print('process time:{:.0f} seconds'.format(spent))
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def main(data_file, # = 'sqlite:///Messages.db',
         file_name, # = 'classifier.pkl',
         remove_cols = False,
         tree_type = True,
         C = 2.0,
         learning_rate = 0.5,
         n_estimators = 80,
         test_size = 0.25,
         best_10 = True,
         pre_tokenize = False,
         condense = True,
         grid_search = False,
         verbose = False):
    '''This is the main Machine Learning Pipeline function. It calls the other 
    ones, in the correct order.
    Example: python train_classifier.py
    Basic parameters:
      - data_file (mandatory) - just indicate the complete path after the command
        Example: python train_classifier.py ../data/Database.db
      - file_name (mandatory) - you need to indicate both data_file and file_name
        Example: python train_classifier.py ../data/Database.db other.pkl
    Extra parameters:
      here you need to indicate both data_file and classifier, in order to use 
      them you can use only one, or more, in any order
      -a -> run metrics over ALL labels (not recommended!) 
            default=False - run metris over the 10 main labels only
      -c -> C parameter for your Classificer (default=2.0)
      -e -> NOT remove duplicates (condense) tokens on a document. Sometimes it 
            turns easier for the Classifier to fit best parameters, others not 
            (default=True - remove duplicates)
      -g -> perform Grid Search over Adaboost before training for the best
            parameters. Please use it wisely, as it costs a lot of processing
            time!
      -l -> learning rate for Adaboost Classifier. It have a tradeoff with
            n_estimators, so consider to tune both parameters (default=0.5)
      -n -> number of maximum estimators for Adaboost (default=80)
      -p -> pre_tokenize - keep preprocessing tokenization column, for saving
            processing time. Obsservation: keeping this column turns the system 
            faster, but may cause instability on Classifier training on Flask
            due to "pipeline leakage" (not recomended) (default=False)  
      -r -> remove columns - if you want to remove (un)trainable columns from
            your y-labels dataset (default=False)
      -s -> change Classifier from Adaboost (tree-type) to LSVM 
            (support vector machine-type)
      -t -> test size for splitting your data (default=0.25)
      -v -> verbose - if you want some verbosity during the running
            (default=False)
      Example: python train_classifier data.db other.pkl -c=0.5 -t=0.2 -r -v
    '''
    run_pipeline(data_file=data_file,
                 file_name=file_name,
                 remove_cols=remove_cols,
                 tree_type=tree_type,
                 C=C,
                 learning_rate=learning_rate,
                 n_estimators=n_estimators,
                 test_size=test_size,
                 best_10=best_10,
                 pre_tokenize=pre_tokenize,
                 condense=condense,
                 grid_search=grid_search,
                 verbose=verbose)
                                           
#########1#########2#########3#########4#########5#########6#########7#########8              
if __name__ == '__main__':

    #first, try to get the system arguments
    args = sys.argv[1:]
    print('args:', args)

    #second, try to charge with the two main arguments
    #if len(args) == 0: #python
    #    main()
    #elif len(args) == 1: #python sqlite:///Messages.db
    #    main(data_file=args[0])
    if len(args) < 2:
        raise Exception('at least two arguments needed')
    elif len(args) == 2: #python sqlite:///Messages.db classifier.pkl
        main(data_file=args[0],
             classifier=args[1])
    else: #default parameters
        remove_cols = False
        tree_type = True
        C = 2.0
        learning_rate = 0.5
        n_estimators = 80
        test_size = 0.25
        best_10 = True
        pre_tokenize = False
        grid_search = False
        condense = True
        verbose = False
                                                                         
        remain_args = args[2:] #elliminate the two main arg 
        for arg in remain_args:
            comm = arg[:2] #get the command part
            if comm == '-a':
                best_10 = False
            elif comm == '-c':
                C = arg[3:]
            elif comm == '-e':
                condense = False
            elif comm == '-g':
                grid_search = True
            elif comm == '-l':
                learning_rate = arg[3:]
            elif comm == '-n':
                n_estimators = arg[3:]
            elif comm == '-p':
                pre_tokenize=True
            elif comm == '-r':
                remove_cols = True
            elif comm == '-s':
                tree_type = False
            elif comm == '-t':
                test_size = arg[3:]
            elif comm == '-v':
                verbose = True
            else:
                raise Exception('invalid argument')
                
        main(data_file=args[0],
             file_name=args[1],
             remove_cols=remove_cols,
             tree_type=tree_type,
             C=C,
             learning_rate=learning_rate,
             n_estimators=n_estimators,
             test_size=test_size,
             best_10=best_10,
             pre_tokenize=pre_tokenize,
             condense=condense,
             grid_search=grid_search,
             verbose=verbose)