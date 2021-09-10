import sys
import pandas as pd
from time import time
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import math
import udacourse2

#########1#########2#########3#########4#########5#########6#########7#########8
def load_data(messages_filepath, 
              categories_filepath,
              messages_index='id',
              categories_index='id',
              verbose=False):
    '''This function loads the data to be processed
    Inputs:
      - messages_filepath (mandatory) - String containing the full path for the
        messages data (in .csv format)
      - categories_filepath (mandatory) - String containing the full path for 
        the categories data (in .csv format)
      - verbose (optional) - if you needed some verbosity, turn it on - Boolean 
        (default=False)
    Output:
      - a merged Pandas Dataset
    '''
    if verbose:
        print('###load_data function started')
    
    begin = time()
    
    #load datasets
    messages = udacourse2.fn_read_data(filepath=messages_filepath, 
                                       index_col=messages_index, 
                                       verbose=verbose)
    categories = udacourse2.fn_read_data(filepath=categories_filepath, 
                                         index_col=categories_index, 
                                         verbose=verbose)
        
    #pre-filter - cleaning duplicates from messages
    #why so earlier? because I turn my dataset lighter, saving time
    #why only messages? because then I can use a Left joint
    #Step I - check number of duplicated messages
    if verbose:
        duplicated = messages[messages.duplicated()].shape[0]
        print('{} duplicated messages has been found'.format(duplicated))
    #Step II - drop index duplicated messages (completely disposable!)
    #https://stackoverflow.com/questions/35084071/concat-dataframe-reindexing-only-valid-with-uniquely-valued-index-objects
    messages = messages.loc[~messages.index.duplicated(keep='first')]
    #Step IIb - drop row duplicates
    messages = messages.drop_duplicates()
    #Step III - check the number of remaining duplicates
    duplicated = messages[messages.duplicated()].shape[0]
    if verbose:
        print('duplicated dropped by index and by row, remaining {} duplicated rows'.format(duplicated))
    if duplicated > 0:
        if verbose:
            print('messages dataframe remains with {} rows'.format(messages.shape[0]))
        raise Exception('something went wrong, messagea remains with duplicated rows')
        
    #merging datasets using SQL-like notation
    #https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
    df = pd.merge(messages, 
                  categories, 
                  left_index=True, 
                  right_index=True, 
                  how='left')
    
    #later adjusts aftter merging
    df = df.set_index('id_x')
    df = df.drop(columns='id_y')
    df.index.name = 'id'
    
    if verbose:
        print('merged dataset has {} rows and {} columns'.format(df.shape[0], 
                                                                 df.shape[1]))
        #print(df.head(5))
            
    spent = time() - begin
    if verbose:
        print('load_data spent time: {:.1f}s ({}min, {:.4f}sec)'.format(spent, 
              math.trunc((spent)/60), math.fmod(spent, 60)))
 
    return df

#########1#########2#########3#########4#########5#########6#########7#########8
def clean_data(df,
               cat_sepparation=';',
               verbose=False):
    '''This function clears the data from the merged dataset
    Input:
      - df - Pandas Dataset containing the data to be cleared
      - verbose (optional) - if you needed some verbosity, turn it on - Boolean 
        (default=False)
    Output:
      - Pandas Dataset cleared
    '''
    if verbose:
        print('###clean_data function started')              
    
    begin = time()
              
    #Step I - split categories into separate category columns          
    cat_seq = df['categories'].iloc[0]
    categories = cat_seq.split(sep=cat_sepparation)
    #this helped this step
    #https://stackoverflow.com/questions/42049147/convert-list-to-pandas-dataframe-column
    categories = pd.DataFrame({'categories': categories})
    #if verbose:
    #    print('categories:',categories.head())
    
    #Step II - create categories column names
    category_colnames = [cat[:-2] for cat in categories['categories']]
    if verbose:
        print('filtered category names for new columns:', category_colnames)
   
    #Step IIb - an alert column flag, for rows that don´t have any category
    df['if_blank'] = False
              
    #Step III - adding new columns, filled with zero value
    for colname in category_colnames:
        df[colname] = 0
    if verbose:
        print('new dataset columns number:',df.shape[1])
        
    #just append the two missing originals columns for treatment
    filtered_cols = category_colnames
    filtered_cols.append('categories')
    filtered_cols.append('if_blank')
    
    #calling the completion function (the treatment!)       
    df[filtered_cols] = df[filtered_cols]\
    .apply(lambda x: udacourse2.fn_test(x, 
                                        verbose=verbose), 
                                        axis=1)
              
    #dropping old categories column
    df = df.drop('categories', axis=1)
    #if verbose:
    #    print(df.head(5))
              
    #check for critical NaN values
    if udacourse2.fn_check_NaN(df, verbose=verbose):
        if verbose:
            print('dataset was checked for NaN and none critical was found')
    else:
        raise Exception('something bad happened with the function check_NaN')
        
    spent = time() - begin
    if verbose:
        print('clean_data elapsed time: {:.1f}s ({}min, {:.4f}sec)'.format(spent, 
              math.trunc((spent)/60), math.fmod(spent, 60)))
    
    return df

#########1#########2#########3#########4#########5#########6#########7#########8
def save_data(df, 
              database_filename, #='sqlite:///Messages.db',
              verbose=False,
              add_report=False):
    '''This function
    Inputs:
      - df (mandatory) - dataframe to be saved on SQLite 
      - database_filename (mandatory) - text string 
      - verbose (optional) - if you need some verbosity, turn it on - Boolean 
        (default=False)
      - add_report (optional) - if you need an additional report about data, 
        turn it on - Boolean (default=False)
    Output:
      - returns True if everything runs OK
    '''
    if verbose:
        ('###save_data function started')
              
    begin = time()         
              
    database = create_engine(database_filename, 
                             poolclass=NullPool) #, echo=True)
    connection = database.connect()
    #attempt to save my dataframe to SQLite
    try:
        df.to_sql('Messages', 
                  database, index=False, 
                  if_exists='replace')
    except ValueError:
        print('something went wrong when was writing data do SQLite')
    
    connection.close()          

    if add_report:
        print('###additional report for the data processed###')
        print('total of columns:', df.shape[1])
        print('valid rows:', df.shape[0])
        
        #counting
        print()
        print ('New columns counting:')
        total = df.shape[0]
        
        #if blank column
        field = 'if_blank'
        count = udacourse2.fn_count_valids(dataset=df, 
                                           field=field, 
                                           criteria=True,
                                           verbose=verbose)
        percent = 100. * (count / total)
        print('{} column: {} ({:.1f}%)'.format(field, count, percent))
        
        print()
        print('Valid labels counting:')
        udacourse2.fn_labels_report(dataset=df)
        
        #columns count
        #expand_list = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 
        #'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 
        #'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 
        #'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 
        #'earthquake', 'cold', 'other_weather', 'direct_report']

        #total = df.shape[0]
        #counts = []

        #for field in expand_list:
        #    count = fn_count_valids(df=df, field=field)
        #    percent = 100. * (count / total)
        #    counts.append((count, field, percent))
        #    print('{}: {} ({:.1f}%)'.format(field, count, percent))

    spent = time() - begin
    if verbose:
        print('saved_data elapsed time: {:.1f}s ({}min, {:.4f}sec)'.format(spent, 
              math.trunc((spent)/60), math.fmod(spent, 60)))
              
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def run_etl(messages_filepath, #='messages.csv', 
            categories_filepath, #='categories.csv',
            database_filepath, #='sqlite:///Messages.db',
            messages_index='id',
            categories_index='id',
            cat_sepparation=';',
            verbose=False,
            add_report=True):
    '''This is the main ETL function. It calls the other ones, in the correct
    order. To call it use: 
    Input:
      - messages_filepath (mandatory) - textt string
      - categories_filepath (mandatory) - text string
      - database_filepath (mandatory) - text string
      - messages_index (optional) - (default='id')
      - categories_index (optional) - (default='id')
      - cat_sepparation (optional) - (default=';')
      - verbose (optional) - if you need some verbosity, turn it on - Boolean 
        (default=False)
      - add_report (optional) - if you need to add a report on save data - 
        Boolean (default=False) 
    '''
    if verbose:
        print('###main system function started')
    
    gen_begin = time()
    
    sys.argv = '1234' #remove later!
                  
    if len(sys.argv) == 4:
        
        #uncomment later!
        #messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, 
                       categories_filepath,
                       messages_index,
                       categories_index,
                       verbose=verbose)

        print('Cleaning data...')
        df = clean_data(df,
                        cat_sepparation,
                        verbose=verbose)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        result = save_data(df, 
                           database_filepath, 
                           verbose=verbose,
                           add_report=add_report)
        if result:
            if verbose:
                print('transaction done')
        else:
            raise Exception('something went wrong with save_data function')
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
              
    spent = time() - gen_begin
    if verbose:
        print('saved_data elapsed time: {:.1f}s ({}min, {:.4f}sec)'.format(spent, 
              math.trunc((spent)/60), math.fmod(spent, 60)))
        
    return True

#########1#########2#########3#########4#########5#########6#########7#########8              
def main(messages_filepath, #='messages.csv',
         categories_filepath, #='categories.csv',
         database_filepath, #='sqlite:///Messages.db',
         messages_index='id',
         categories_index='id',
         cat_sepparation=';',
         verbose=False,
         add_report=False):
    '''This is the main ETL Pipeline function. It calls the other 
    ones, in the correct order.
    Example: python process_data.py
      Basic parameters:
      - messages_filepath - just indicate the complete path after the command
        Example: python process_data.py msgs.csv
      - categories_filepath - you need to indicate both data_file and classifier
        Example: python process_data.py msgs.csv cats.csv
      - database_filapath - you need to indicate all the three steps
        Example: python process_data.py msgs.csv cats.csv response.db
      Extra parameters
      you need to indicate messages, categories and database, in order to use 
      them. You can use only one, or more, in any order
      -m -> messages_index - if you want to alter (default='id')
      -c -> categories_index - if you want to alter (default='id')
      -s -> cat_sepparation - if you want to alter (default=';')
      -v -> verbose - if you want verbosity (default=False)
      -a -> add_report - if you want adittional report (default=False)
      Examples: python process_data msgs.csv resp.db -m=ID -c=index -s=; -v -a
      python process_data.py messages.csv disaster_categories.csv DisasterResponse.db
      
    '''
    run_etl(messages_filepath=messages_filepath, 
            categories_filepath=categories_filepath,
            database_filepath=database_filepath,
            messages_index=messages_index,
            categories_index=categories_index,
            cat_sepparation=cat_sepparation,
            verbose=verbose,
            add_report=add_report) 

if __name__ == '__main__':
    #parse sys.argv[1:] using optparse or argparse or what have you
    #https://stackoverflow.com/questions/14500183/in-python-can-i-call-the-main-of-an-imported-module
    #main(foovalue, barvalue, **dictofoptions)
    
    #first, try to get the system arguments
    args = sys.argv[1:]
    print('args:', args)

    #second, try to charge with the three main arguments
    #if len(args) == 0: #python process_data
    #    main()
    #elif len(args) == 1: #python process_data messages.csv
    #    main(messages_filepath=args[0])
    #elif len(args) == 2: #python process_data messages.csv categories.csv
    #    main(messages_filepath=args[0],
    #         categories_filepath=args[1])
    if len(args) < 3:
        raise Exception('minimum: 3 fundamental arguments needed to run')
    elif len(args) == 3: #ython process_data messages.csv categories.csv sqlite:///Messages.db
        main(messages_filepath=args[0],
             categories_filepath=args[1],
             database_filepath=args[2])
    else:
        messages_index='id'
        categories_index='id'
        cat_sepparation=';'
        verbose=False
        add_report=False

        remain_args = args[3:] #three main args    
        for arg in remain_args:
            comm = arg[:2] #get the command part
            if comm == '-m':
                messages_index = arg[3:]
            elif comm == '-c':
                categories_index = arg[3:]
            elif comm == '-s':
                cat_sepparation = arg[3:]
            elif comm == '-v':
                verbose=True
            elif comm == '-a':
                add_report=True
            else:
                raise Exception('invalid argument')
                
        main(messages_filepath=args[0], #full calling
             categories_filepath=args[1],
             database_filepath=args[2],
             messages_index=messages_index,
             categories_index=categories_index,
             cat_sepparation=cat_sepparation,
             verbose=verbose,
             add_report=add_report)