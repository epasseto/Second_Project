import sys
import pandas as pd
from time import time
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import math

###Main Functions (04)##########################################################
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
    messages = fn_read_data(filepath=messages_filepath, 
                            index_col=messages_index, 
                            verbose=verbose)
    categories = fn_read_data(filepath=categories_filepath, 
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
    df[filtered_cols] = df[filtered_cols].apply(lambda x: fn_test(x, 
                                                                  verbose=verbose), 
                                                                      axis=1)
              
    #dropping old categories column
    df = df.drop('categories', axis=1)
    #if verbose:
    #    print(df.head(5))
              
    #check for critical NaN values
    if fn_check_NaN(df, verbose=verbose):
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
              database_filename='sqlite:///Messages.db',
              verbose=False,
              add_report=False):
    '''This function
    Inputs:
      - df (mandatory) - dataframe to be saved on SQLite 
      - database_filename (optional) - (default='sqlite:///Messages.db') 
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
        count = fn_count_valids(df=df, field=field, criteria=True)
        percent = 100. * (count / total)
        print('{} column: {} ({:.1f}%)'.format(field, count, percent))
        
        #columns count
        expand_list = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 
       'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 
       'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 
       'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 
       'earthquake', 'cold', 'other_weather', 'direct_report']

        total = df.shape[0]
        counts = []

        for field in expand_list:
            count = fn_count_valids(df=df, field=field)
            percent = 100. * (count / total)
            counts.append((count, field, percent))
            print('{}: {} ({:.1f}%)'.format(field, count, percent))

    spent = time() - begin
    if verbose:
        print('saved_data elapsed time: {:.1f}s ({}min, {:.4f}sec)'.format(spent, 
              math.trunc((spent)/60), math.fmod(spent, 60)))
              
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def main(messages_filepath, 
         categories_filepath,
         database_filepath,
         messages_index='id',
         categories_index='id',
         cat_sepparation=';',
         verbose=False,
         add_report=True):
    '''This is the main ETL function. It calls the other ones, in the correct
    order. To call it use: 
    Input:
      - messages_filepath (mandatory) - string - (default='messages.csv') 
      - categories_filepath (mandatory) - string - (default='categories.csv')
      - database_filepath (mandatory) - string - (default='sqlite:///Messages.db')
      - messages_index (optional) - (default='id')
      - categories_index (optional) - (default='id')
      - cat_sepparation (optional) - (default=';')
      - verbose (optional) - if you need some verbosity, turn it on - Boolean 
        (default=False)
      -add_report (optional) - 

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
if __name__ == '__main__':
    main()

###System Subfunctions (in alfabetic order) (05)################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_check_NaN(df, 
                 verbose=False):
    '''This function checks your final dataframe for critical NaN
    (column 'original' may have NaN)
    Inputs:
        df (mandatory) - Pandas Dataframe to be checked
        verbose (optional) - if you want some verbosity (default=False)
    Output:
        returns True if the function runned well
    '''
    if verbose:
        print('*check for NaN subfunction started')
    
    for column in df.columns:
        result = df[column].isnull().values.any()
        if result:
            if verbose:
                print('column: {} -> {}'.format(column, result))
            if column == 'original':
                if verbose:
                    print('*original can have NaN, as it refers to foreign languages posting (some cases only)')
            else:
                raise Exception('some critical rows with NaN were found in your dataframe')
                
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_count_valids(df, 
                    field, 
                    criteria=1, 
                    verbose=False):
    '''This function counts column data for additional report of saved_data
    Inputs:
      - df (mandatory) - dataset for stats
      - field (mandatory) - field to be counted
      - criteria (optional) - valid cound criteria (default=1)
      - verbose (optional) - if you want some verbosity (default-False)
    *Comment: this function was not optimized! It takes only one column per
     calling. As it is only an optional report that probably will turned off in
     a prodution system, it don´t matter too much. For future improvement,
     please consider multicolumns processing for design a new count_valids
     function
    '''
    if verbose:
        print('*counting subfunction started')
    
    return df[field][df[field] == criteria].sum()

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_read_data(filepath, 
                 index_col='id',
                 verbose=False):
    '''This function reads a .csv file
    Inputs:
      - filepath (mandatory) - String containing the full path for the data to
        oppened
      - index_col (optional) - String containing the name of the index column
        (default='id')
      - verbose (optional) - if you needed some verbosity, turn it on - Boolean 
        (default=False)
    Output:
      - Pandas Dataframe with the data
    '''
    if verbose:
        print('*subfunction read_data started')
    
    #reading the file
    df = pd.read_csv(filepath)
    df.set_index(index_col)
    
    if verbose:
        print('file readed as Dataframe')

    #testing if Dataframe exists
    #https://stackoverflow.com/questions/39337115/testing-if-a-pandas-dataframe-exists/39338381
    if df is not None: 
        if verbose:
            print('dataframe created from', filepath)
            #print(df.head(5))
    else:
        raise Exception('something went wrong when acessing .csv file', filepath)
    
    #setting a name for the dataframe (I will cound need to use it later!)
    ###https://stackoverflow.com/questions/18022845/pandas-index-column-title-or-name?rq=1
    #last_one = filepath.rfind('/')
    #if last_one == -1: #cut only .csv extension
    #    df_name = filepath[: -4] 
    #else: #cut both tails
    #    df_name = full_path[last_one+1: -4]   
    #df.index.name = df_name
    #if verbose:
    #    print('dataframe index name setted as', df_name)

    return df

#########1#########2#########3#########4#########5#########6#########7#########8              
def fn_test(x, verbose=False):
    if verbose:
        print('###')      
    string_cat = x['categories']
    #at least one case found
    if string_cat.find('1') != -1:
        #break into components
        alfa = set(string_cat.split(sep=';'))
        #loop on components
        for beta in alfa:
            if beta.find('1') != -1:
                if verbose:
                    print(beta[:-2])
                gama = beta[:-2]
                x[gama] = 1
    #no cases!
    else:
        if verbose:
            print('*empty element*')
        x['if_blank'] = True
                      
    return x