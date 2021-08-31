<<<<<<< HEAD
#import matplotlib.patches as mpatches
#import matplotlib.patches as mpatches
#import matplotlib.style as mstyles
#import matplotlib.pyplot as mpyplots
#from matplotlib.figure import Figure
#import seaborn as sns

import re
import pandas as pd
from time import time
import numpy as np
import math #sorry, I need both!
import statistics

#natural language toolkit
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

#metrics for Classifiers
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_cat_condenser(subset,
                     name='df',
                     opperation='all_sub',
                     verbose=False):
    '''this function...
    Inputs:
      - subset (mandatory) - the subset of "related" supercategory to be\
        processed. Possible options: 'aid', 'wtr', 'ifr' and 'main'.
      - name (optional) - the name of the dataset (default='df')
      - opperation (optional) - choose one opperation
        - 'all_subs' (default) - for taking all sub sets
        - 'sub_not_main'- filter for taking subs, with not main setted
        - 'main_not_sub' - filter for main setted, but none sub
      - verbose (optional) - if you want some verbosity (default=False)
    Outputs:
      - in a form of a Tuple, containing 3 items, as
      - Tuple[0] - the filtering statement, as a text string
      - Tuple[1] - the set name, as a text string
      - Tuple[2] - the list of subsets, as a Python list
    This function can be lately expanded for:
      - automatically selecting a Boolean crieteria for multi-filtering
      - including the "~" (not) statement just before the filter condition
      - allowing Integer entries for element, as 1
      - verifying consistencies (e.g., condition can be only "^" or "&"
    '''
    if verbose:
        print('###function cat_condenser started')
    
    begin = time()
        
    #paramameters for processing
    opperator='==' 
    super_item = 'related'
    #other_super = ['request', 'offer', 'direct_report']
    if opperation == 'empty_sub':
        element = '0' 
        condition = '&'
    else:
        element = '1' 
        condition = '^'    

    if subset == 'aid':
        set_item = 'aid_related'
        sub_lst = ['food', 'shelter', 'water', 'death', 'refugees', 'money', 
                   'security', 'military', 'clothing', 'tools', 'missing_people', 
                   'child_alone', 'search_and_rescue', 'medical_help', 
                   'medical_products', 'aid_centers', 'other_aid']
    elif subset == 'wtr':
        set_item = 'weather_related'
        sub_lst = ['earthquake', 'storm', 'floods', 'fire', 'cold', 
                   'other_weather']
    elif subset == 'ifr':
        set_item = 'infrastructure_related'
        sub_lst = ['buildings', 'transport', 'hospitals', 'electricity', 
                   'shops', 'other_infrastructure']
    elif subset == 'main':
        set_item = 'related'
        sub_lst = ['aid_related', 'weather_related', 'infrastructure_related']
    else:
        raise Exception('invalid category for subset')
            
    out_str = fn_create_string(sub_lst=sub_lst,
                               dataset=name,
                               opperator=opperator,
                               element=element,
                               condition=condition,
                               verbose=verbose)        
        
    if opperation == 'all_sub':
        if verbose:
            print('processing for subset ended')
        output = out_str
    elif opperation == 'sub_not_main':
        if verbose:
            print('processing for sub_not_main ended')
        output = "(" + name + "['" + set_item + "'] == 0) & (" + out_str + ")"
    elif opperation == 'main_not_sub':
        if verbose:
            print('processing for main_not_sub ended')
        output = "(" + name + "['" + set_item + "'] == 1) & ~(" + out_str + ")"
    elif opperation == 'empty_sub':
        if verbose:
            print('processing for empty subset ended')
        output = "(" + name + "['" + set_item + "'] == 1) & (" + out_str + ")"
    else:
        raise Exception('category is invalid')

    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))

    return (output, set_item, sub_lst)

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_count_valids(dataset, 
                    field, 
                    criteria=1, 
                    verbose=False):
    '''This function count all valids for a field in a dataset
    Inputs:
      - dataset (mandatory) - the dataset to be processed
      - field (mandatory) - the field to be counted
      - criteria (optional) - what counts as a valid one (defauld=1)
      - verbose (optional) - if you want some verbosity (default=False)
    Output:
      - number of valid counts (Integer)
    '''
    if verbose:
        print('###counting function initiated')

    begin = time()  

    result = dataset[field][dataset[field] == criteria].sum()
        
    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
        
    return result

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_string(sub_lst,
                     dataset,
                     opperator,
                     element,
                     condition,
                     verbose):
    '''This function creates a string for filtering dataset columns
    Inputs:
      - sub_lst (mandatory) - the list of subcategories for the main category
        (Python List)
      - dataset (mandatory) - the name of the dataset to be filtered (String)
      - opperator (mandatory) - the opperator for filtering (String Char) 
      - element (mandatory) - the element for filtering (String)
      - condition (mandatory) - the condition for filtering (string Char)
      - verbose (optional) - if you want some verbosity (default=False) 
    Output: filtering string for dataset opperations (String)
    '''
    if verbose:
        print('###function create_string started')
    
    begin = time()
    string = ''

    for item in sub_lst:
        string = string + "(" + dataset + "['" + item + "'] " + opperator + \
" " + element + ")" + " " + condition + " "
        
    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))

    return string[:-3]

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_getKey(item):
    '''This is an elementary function for returning the key from an item 
    from a list
    Input:
      - an item from a list
    Output it´s key value
    It is necessary to run the ordered list function
    '''
    return item[0]

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_group_check(dataset,
                   subset,
                   correct=False,
                   shrink=False,
                   shorten=False,
                   verbose=False):
    '''This funtion calls subfunctions, with the correct parameters, in order
    to check a group of subcategories, given a category, according to
    database fundamentals, and pass options as to shrink, reduce or correct
    the dataset.
    Inputs:
      - dataset (mandatory) - the dataset to be checked (Pandas dataset)
      - subset (mandatory) - the subset to be checked (text string)
      - correct (optional) - if you want to correct inconsistencies
        (boolean, default=False)
      - shrink (optional) - if you want to shrink dataframe for the 
        focused dataset (boolean, default=False)
      - shorten (optional) - filter rows just for the main group
        (boolean, default=False)
      - verbose (optional) - if you want some verbosity (default=False)
    Output:
      - corrected dataframe, for database theory about inconsistencies
    *future implementation, correct & shrink for other criteria than 1*
     (implemented!)
    '''
    if verbose:
        print('###function group_check started')
    
    begin = time()
    
    #retrieving the name of the dataset, or using a default name
    #try:
    #    dataset.name
    #    name = dataset.name
    #except AttributeError:
    #    name = 'df'
    name = 'dataset'
    
    ###test for main class counting
    #I need this early calling, justo to get the main set name
    mainotsub = fn_cat_condenser(subset=subset,
                                 name=name,
                                 opperation='main_not_sub')
    main_class = mainotsub[1]
    count_main = dataset[dataset[main_class] == 1].shape[0]
    if verbose:
        print('  - count for main class:{}, {} entries'.\
format(main_class, count_main))
        
    ###test for main, without any sub-category 
    count_mainotsub = dataset[eval(mainotsub[0])].shape[0]
    
    if verbose:
        print('  - for main, without any sub-categories,  {} entries'.\
format(count_mainotsub))
        
    ###test for subcategories counting
    count_subs = dataset[eval(fn_cat_condenser(subset=subset,
                                               name=name,
                                               opperation='all_sub')[0])].shape[0]
    if verbose:
        print('  - for subcategories,  {} entries'.format(count_subs))

    ###test for sub, without main registered (data inconsistency)
    subnotmain = fn_cat_condenser(subset=subset,
                                  name=name,
                                  opperation='sub_not_main')
    count_subnotmain = dataset[eval(subnotmain[0])].shape[0]
    if verbose:
        print('  - for lost parent sub-categories,  {} entries'.\
format(count_subnotmain))
        
    if correct:
        #correcting to 1 - future: other criteria
        dataset.loc[dataset[eval(subnotmain[0])].index, subnotmain[1]] = 1
        #checking the correction
        subnotmain = fn_cat_condenser(subset=subset,
                                      name=name,
                                      opperation='sub_not_main')
        count_subnotmain = dataset[eval(subnotmain[0])].shape[0]
        if verbose:
            print('    *correcting, new count: {} entries'.\
format(count_subnotmain))
            
    if shrink:
        new_cols = ['message', 'genre', 'if_blank']
        new_cols.append(subnotmain[1]) #append the group column
        new_cols = new_cols + subnotmain[2]
        dataset = dataset[new_cols] #shrink for selected columns
        if verbose:
            print('    *shrinking, dataset now have: {} columns'.\
format(dataset.shape[1]))
            
    if shorten: #future:create other criteria
        dataset = dataset[dataset[subnotmain[1]] == 1]
        if verbose:
            print('    *shortening, dataset now have: {} lines'.\
format(dataset.shape[0]))
    
    end = time()
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))

    return dataset

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_labels_report(dataset, 
                     max_c=False, 
                     verbose=False):
    '''This is a report only function!
    Inputs:
      - dataset (mandatory) - the target dataset for reporting about
      - max_c (optional) - maximum counting - if you want to count for all elements,
        set it as False - (default=False)
      - verbose (optional) - if you want some verbosity (default=False)
    Output:
      - no output, shows reports about the labels counting
    '''
    begin = time()
    
    expand_lst = ['related', 'request', 'offer', 'aid_related', 
                  'infrastructure_related', 'weather_related', 
                  'direct_report']
    
    aid_lst = ['food', 'shelter', 'water', 'death', 'refugees', 'money', 
               'security', 'military', 'clothing', 'tools', 'missing_people', 
               'child_alone', 'search_and_rescue', 'medical_help', 
               'medical_products', 'aid_centers', 'other_aid']
    
    weather_lst = ['earthquake', 'storm', 'floods', 'fire', 'cold', 
                   'other_weather']
    
    infrastructure_lst = ['buildings', 'transport', 'hospitals', 'electricity', 
                          'shops', 'other_infrastructure']
    
    expand_list = expand_lst + aid_lst + weather_lst + infrastructure_lst
    total = dataset.shape[0]
    counts = []

    #count for labels - not yet ordered!
    for field in expand_list:
        count = fn_count_valids(dataset=dataset, field=field)
        percent = 100. * (count / total)
        counts.append((count, field, percent))
        #if verbose:
        #    print('{}:{} ({:.1f}%)'.format(field, count, percent))
        
    #sort it as sorted tuples
    sorted_tuples = sorted(counts, key=fn_getKey, reverse=True)

    i=1
    c=2

    for cat in sorted_tuples:
        count, field, percent = cat
        print('{}-{}:{} ({:.1f}%)'.format(i, field, count, percent))
        if max_c:
            if c > max_c:
                break
        i += 1
        c += 1

    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
        
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_subcount_lists(column, 
                      verbose=False):
    '''This function takes a column that have a list, iterate them and 
    count unique items for each registry.
    The objective is to count different individuals that are nested.
    It also returns the sum for empty lists, it they exist.
    Inputs:
      - col (mandatory) - the column containing a list to be harshed (Pandas 
        Series)
      - verbose (optional) - it is needed some verbosity, turn it on - 
        (Boolean, default=False)
    Output:
      - a Dictionnary with the counting for each item, plus the number of rows 
        with NaN
    Obs: this is an adaptation from the fn_subcount_cols, from my authory, for 
         another Udacity project. The original version takes data from columns
         as string of characters, each element sepparated by ";". Now it takes
         lists, already sepparated.
    '''
    begin = time()
        
    #I already know that I want these entries, even if they finish as zero!    
    items_dic = {'empty_lists': 0,
                 'valid_rows': 0}
    harsh_dic = {} #temporary dictionnary for harshing
    
    for list in column:     
        if len(list) == 0: #ampty list (could be dangerous for our project!)
            if verbose:
                print('*empty list!')
            items_dic['empty_lists'] += 1
        else:
            #It may be necessary to remove all spaces inside the harshed item
            #I found the best way to do this at Stack Overflow, here:
            #https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string
            if verbose:
                print('splitted registry:', list)
            items_dic['valid_rows'] += 1
            
            for element in list:
                if verbose:
                    print('element for accounting:', element)
                if element in harsh_dic:
                    harsh_dic[element] += 1
                else:
                    harsh_dic[element] = 1

    #Why I made this strange sub-dictionnary insertion?
    #So, I think of a kind of Json structure will be really useful for my 
    #Udacity miniprojects (Yes, I am really motivated to study Json... it looks 
    #nice for my programming future!)
    items_dic['elements'] = harsh_dic
    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
        print('*************')
        print('dictionnary of counting items created:')
        print(items_dic)
    
    return items_dic

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_test(x, 
            verbose=False):
    '''This function tests for empty elements, and splits a unique categorie
    into counting "1" for valid new coluns, under the item list
    Inputs:
      - x (mandatory) - a row from a iterated Dataset, containing all the rows
        for (Pandas Series)
        - "categories" register is the one that contains no element, or a list
          of the rows to be valid as "1"
        - "if_blank" register receives "1" only if "categories" is an empty list
        - the other registers are the canditates to receive a "1" (initial set
          is 0)
      - verbose (optional) - if you want some verbosity (default=False)
    Output:
      - x (Pandas Series) containing the processing expansion of "categories" 
        into columns for your dataset
    '''
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

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_tokenize(msg_text, 
                lemmatize=True, 
                rem_city=False,
                agg_words=False,
                rem_noise=False,
                elm_short=False,
                unhelpful_words=[],
                great_noisy=False,
                verbose=False):
    """This functions turns brute messages into tokens, for Machine Learning 
    training
    Inputs:
      - msg_text - string (mandatory) - a text string (not too long), as a 
        Tweeter message
      - lemmatize - boolean (optional) - if you want to run lemmatizer over 
        tokenizer, please
        turn it on (default=False)
      - rem_city - boolean (optional) - list of cities > than 100.000 
        inhabitants, to remove from
        messages (default=False) 
      - verbose - boolean (optional) - if you need some verbosity, turn it on 
        (default=False) 
    Output:
      - a list of tokens (reduced meaningful words)
    New addictions, ver 1.1:
      - built an aggregate function, to prevent duplicate words on a tokenized 
        string (as ['food', ... 'food', ...])
      - built a unnuseful words list, to remove common communication words, as 
        'thanks' and other noisy words for Machine Learning training
      - built an ellimination for too short words
    New Imputs:
      - agg_words - boolean (optional) - if you want to aggregate a list as a 
        set and turning back into a list (default=False)
      - rem_noise - boolean (optional) - if you want to remove the words from a 
        predefined list
        (default=False)
      - unhelpful_words - list (optional) - if you want to provide your own 
        noise_words list, ativate it
        (default = [])
      - elm_short = boolean/integer (optional) - if you want to elliminate 
        words shorter than a number please provide a number (e.g. 3)
        (default=False)
    """
    #if verbose:
    #    print('###Tokenizer function started')
        
    if rem_city:
        #print('*special list for city removal is loading')
        df_countries = pd.read_csv('all.csv')
        df_countries = df_countries['name'].apply(lambda x: x.lower())
        countries = df_countries.tolist()
        
    #add ver 1.1
    if rem_noise and (len(unhelpful_words) == 0):
        unhelpful_words = ['thank', 'thanks', 'god', 'fine', 'number', 'area', 
            'let', 'stop', 'know', 'going', 'thing', 'would', 'hello', 'say', 
            'neither', 'right', 'asap', 'near', 'want', 'also', 'like', 'since', 
            'grace', 'congratulate', 'situated', 'tell', 'almost', 'hyme', 
            'sainte', 'croix', 'ville', 'street', 'valley', 'section', 'carnaval',
            'rap', 'cry', 'location', 'ples', 'bless', 'entire', 'specially', 
            'sorry', 'saint', 'village', 'located', 'palace', 'might', 'given', 
            'santo', 'jesus', 'heart', 'sacred', 'please', 'named', 'janvier', 
            'avenue', 'tinante', 'cross', 'miracle', 'street', 'abroad', 'someone', 
            'country', 'rue']
        #if verbose:
        #    print('*{} added words on noise filter'.format(len(unhelpful_words)))
    
    #add ver 1.2
    if great_noisy:
        noisy_words = ['people', 'help', 'need', 'said', 'country', 'government', 
            'one', 'year', 'good', 'day', 'two', 'get', 'message', 'many', 'region', 
            'city', 'province', 'road', 'district', 'including', 'time', 'new', 
            'still', 'due', 'local', 'part', 'problem', 'may', 'take', 'come', 
            'effort', 'note', 'around', 'person', 'lot', 'already', 'situation', 
            'see', 'response', 'even', 'reported', 'caused', 'village', 'bit', 
            'made', 'way', 'across', 'west', 'never', 'southern', 'january', 
            'least', 'zone', 'small', 'next', 'little', 'four', 'must', 'non', 
            'used', 'five', 'wfp', 'however', 'com', 'set', 'every', 'think', 
            'item', 'yet', 'carrefour', 'asking', 'ask', 'site', 'line', 'put', 
            'unicef', 'got', 'east', 'june', 'got', 'ministry', 'http', 'information', 
            'area', 'find', 'affected', 'relief', 'well', 'million', 'give','state', 
            'send', 'team', 'three', 'make', 'week', 'santiago', 'service', 'official', 
            'month', 'united', 'nation', 'world', 'provide', 'report', 'much', 
            'thousand', 'call', 'level', 'prince', 'organization', 'agency', 
            'according', 'another', 'along', 'back', 'morning', 'news', 'town', 
            'centre', 'long', 'answer', 'management', 'main', 'crisis', 'delmas', 
            'tuesday', 'department', 'end', 'others', 'etc', 'among', 'general', 
            'july', 'six', 'past', 'eastern', 'told', 'haitian']
    
    #First step, lowering the case and taking words
    #lowering, you reduce variability
    low_text = msg_text.lower()
    #I need to take only valid words
    #a good Regex can good a nice job on finding and cleaning words
    #I created only a basic one (very common Regex filter) <- enhance it later!
    gex_text = re.sub(r'[^a-zA-Z]', ' ', low_text)
    first_step = len(gex_text)
        
    #Second step, tokenize & remove stop words
    #a tokenizer reduce words for their nearest more common synonym
    col_words = word_tokenize(gex_text)
    #stop words are these that don´t have an intrinsic meaning
    #languages use them for better gramatical construction on phrases
    unnuseful = stopwords.words("english")
    output = [word for word in col_words if word not in unnuseful]
    second_step = len(output)

    #Optional Step, remove cities names form the text
    if rem_city:
        #if verbose:
        #    print('*optional city names removal started')
        output = [word for word in output if word not in countries]
        optional_step = len(output)
    
    #Optional Step, included on ver 1.1
    if rem_noise:
        output = [word for word in output if word not in unhelpful_words]
        optional_step2 = len(output)
    
    #Third step, lemmatize
    #a lemmatizer reduce words for their root form - reduce variability
    #normally they apply both, tokenizer and lemmatizer
    #they area a bit redundant, so we can disable lemmatizer
    if lemmatize:
        output = [WordNetLemmatizer().lemmatize(word) for word in output]
        third_step = len(output)
    
    #add ver 1.1
    if agg_words:
        output = list(set(output))
        agg_step = len(output)
    
    #add ver 1.1
    if elm_short:
        if isinstance(elm_short, int): #test if integer
            output = [word for word in output if len(word) >= elm_short]
            optional_step3 = len(output)
    
    #add ver 1.2
    if great_noisy:
        output = [word for word in output if word not in noisy_words]
        optional_step4 = len(output)
         
    if verbose:
        if rem_city and rem_noise:
            print('Tokens-start:{}, token/stop:{}, remove cities:{} &noise:{}'.\
format(first_step, second_step, optional_step, optional_step2))
        elif rem_city:
            print('Tokens-start:{}, token/stop:{}, remove cities:{}'.\
format(first_step, second_step, optional_step))
        elif rem_noise:
            print('Tokens-start:{}, token/stop:{}, remove noise:{}'.\
format(first_step, second_step, optional_step2))
        else:
            print('Tokens-start:{}, token/stop:{}'.format(first_step, 
                                                          second_step))
        if lemmatize:
            print(' +lemmatizer:{}'.format(third_step))
        if elm_short:
            print(' +eliminate short:{}'.format(optional_step3))
            
        if great_noisy:
            print(' +eliminate noisy from 300:{}'.format(optional_step4))
    
    return output

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_tokenize_fast(msg_text,
                     condense=False,
                     verbose=False):
    """This is the fast version for word tokenizer. It makes only one loop for 
    all the selected as best functions
    Inputs:
      - msg_text - string (mandatory) - a text string (not too long), as a 
        Tweeter message
      - condense - elliminate duplicated tokens from each document (default=False)
      - verbose - boolean (optional) - if you need some verbosity, turn it on 
        (default=False) 
    Output:
      - a list of tokens (reduced meaningful words)
    """
    #if verbose:
    #    print('###Tokenizer function started')
        
    cleared = []
    unnuseful = stopwords.words("english")
    #marked for remove
    unhelpful_words = ['thank', 'thanks', 'god', 'fine', 'number', 'area', 'let', 
        'stop', 'know', 'going', 'thing', 'would', 'hello', 'say', 'neither', 
        'right', 'asap', 'near',  'also', 'like', 'since', 'grace', 'congratulate', 
        'situated', 'ville', 'street', 'valley', 'section', 'rap',  'location', 
        'ples', 'bless', 'entire', 'specially', 'sorry', 'saint', 'village', 
        'located', 'palace', 'might', 'given', 'santo', 'jesus', 'heart', 'sacred', 
        'named', 'janvier', 'avenue', 'tinante', 'cross', 'street', 'abroad', 
        'someone', 'country', 'rue', 'people',  'said', 'country', 'one', 'year', 
        'good', 'day', 'two', 'get', 'message', 'many', 'region', 'city', 'province', 
        'including', 'time', 'new',  'due', 'local', 'part',  'may', 'take', 'come', 
        'note', 'around', 'person', 'lot', 'already',  'see', 'response', 'even', 
        'village', 'bit', 'made', 'way', 'across', 'west', 'never', 'southern', 
        'january', 'least', 'zone', 'small', 'next', 'little', 'four', 'must', 'non', 
        'used', 'five', 'wfp', 'however', 'com', 'set', 'every', 'think', 'item', 
        'yet', 'site', 'line', 'put', 'got', 'east', 'june', 'got', 'ministry', 'http',  
        'area', 'well', 'state', 'send', 'three', 'make', 'week', 'service', 'told',
        'official', 'world', 'much', 'level', 'prince', 'road', 'district', 'main',
        'according', 'another', 'along', 'back',  'town', 'centre', 'long', 'management', 
        'tuesday', 'department', 'end', 'others', 'etc', 'among', 'general', 'july', 
        'imcomprehensibley', 'incomprehensible', 'six', 'past', 'eastern', 'could',
         'previous', 'regards', 'cul', 'pitrea', 'northern']
    #not removing (I´m in doubt about them!)
    in_doubt = ['carrefour', 'delmas', 'cuba', 'haitian', 'haiti','affected', 'relief',
        'problem', 'united', 'nation', 'team', 'provide', 'report', 'million', 'give',
        'santiago', 'month', 'morning', 'news', 'help', 'need', 'cry', 'please', 'still',
        'crisis', 'answer', 'reported', 'caused', 'asking', 'ask', 'thousand', 'information',
        'want', 'call', 'effort', 'situation', 'tell', 'almost', 'hyme', 'sainte', 'croix',
        'miracle', 'unicef', 'find', 'organization', 'agency', 'carnaval', 'government']
    
    #if you want to try both lists for removal, please uncomment the following line
    #unhelpful_words = unhelpful_words + in_doubt     
    
    #lowering, you reduce variability
    low_text = msg_text.lower()

    #take only valid words by Regex
    gex_text = re.sub(r'[^a-zA-Z]', ' ', low_text)
        
    #tokenize & remove stop words
    col_words = word_tokenize(gex_text)
        
    #remove stop words + unhelpful words + lemmatize
    for word in col_words:
        if ((word not in unnuseful) and (word not in unhelpful_words)) and (len(word) >= 3):
            WordNetLemmatizer().lemmatize(word)
            cleared.append(word)
    
    if condense:        
        cleared = list(set(cleared)) #if you don't let repeated tokens
        
    if verbose:
        print(cleared)
             
    return cleared

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_valids_report(dataset, 
                     verbose=False):
    '''This is a report function! It calls the Count Valids function for each
    label of a dataset, and shows a report about it.
    Input:
      - dataset for valid items report (in percentage)
      - verbose (optional) - if you want some verbosity (default=False)
    Output:
      - none
    '''
    print('###function valids_report started')
    begin = time()
    total = dataset.shape[0]
    field = 'if_blank'
    count = fn_count_valids(dataset=dataset, 
                            field=field, 
                            criteria=True)
    percent = 100. * (count / total)
    print('  *{}:{} ({:.1f}%)'.format(field, count, percent))
    
    spent = time() - begin
    if verbose:
        print('process time:{:.4f} seconds'.format(spent))
                
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_scores_report2(y_test, 
                      y_pred,
                      best_10=False,
                      average='binary',
                      verbose=False):
    '''This function tests the model, giving a report for each label in y.
    It shows metrics, for reliable trained labels
    If the label could not be trained, it gives a warning about it 
    Inputs:
      - y_test (mandatory) - the y data for testing
      - y_pred (mandatory) - the y predicted for the model
      - best_10 (optional) - for best 10 scores
      - average (optional) - method for average  - possible uses:
        ‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary'(default)
      - verbose (optional) - if you want some verbosity (default=False)
    Output:
      - none
    '''
    if verbose:
        print('###function scores_report started')
    
    begin = time()
    call_lst = []
    f1_lst = []
    precision_lst = []
    recall_lst = []
    worst_f1 = [1., 'none']
    worst_precision = [1., 'none']
    worst_recall = [1., 'none']
    
    if best_10:
        if verbose:
            print('using top 10 labels')
        col_labels = ['aid_related', 'weather_related', 'direct_report', 'request', 
                      'other_aid', 'food', 'earthquake', 'storm', 'shelter', 'floods']
    else:
        if verbose:
            print('using all the labels')
        col_labels = y_test.columns #iterate with all the labels
        

    #first, create tupples with (name, index)
    for col in col_labels:
        call_lst.append((col, y_test.columns.get_loc(col)))
    
    #second, iterate tupples
    for col_tuple in call_lst:
        #print(col_tuple)
        column = col_tuple[0]
        index = col_tuple[1]
    
        if verbose:
            print('######################################################')
            print('*{} -> label iloc[{}]'.format(col_tuple[0], col_tuple[1]))
        
        #test for untrained column, if passes, shows report
        y_predicted = y_pred[:, index]
        if (pd.Series(y_predicted) == 0).all(): #all zeroes on predicted
            report = "  - as y_pred has only zeroes, report is not valid"
        else:
            report = classification_report(y_test[column], 
                                           y_predicted)
            accuracy = f1_score(y_test[column], 
                                y_predicted,
                                pos_label=1,
                                average=average)
            f1_lst.append(accuracy)
            if accuracy < worst_f1[0]:
                worst_f1[0] = accuracy
                worst_f1[1] = column
            
            precision = precision_score(y_test[column], 
                                        y_predicted, 
                                        pos_label=1, 
                                        average=average)
            precision_lst.append(precision)
            if precision < worst_precision[0]:
                worst_precision[0] = precision
                worst_precision[1] = column
            
            recall = recall_score(y_test[column], 
                                  y_predicted, 
                                  pos_label=1, 
                                  average=average)
            recall_lst.append(recall)
            if recall < worst_recall[0]:
                worst_recall[0] = recall
                worst_recall[1] = column
        if verbose:
            print(report)
     
    try:
        accuracy = statistics.mean(f1_lst)
    except StatisticsError:
        if verbose:
            print('*no valid element for the labels list!')
        return False

    precision = statistics.mean(precision_lst)
    recall = statistics.mean(recall_lst)
    
    if verbose:
        print('###Model metrics for {} labels:'.format(len(f1_lst)))
        print(' Accuracy: {:.3f} ({:.1f}%)'.format(accuracy, accuracy*100))
        print(' Precision: {:.3f} ({:.1f}%)'.format(precision, precision*100))
        print(' Recall: {:.3f} ({:.1f}%)'.format(recall, recall*100))
        print()
        print('###Worst metrics:')
        print(' Accuracy: {:.3f} ({:.1f}%) for {}'.format(worst_f1[0], 
                                                         worst_f1[0]*100,
                                                         worst_f1[1]))
        print(' Precision: {:.3f} ({:.1f}%) for {}'.format(worst_precision[0], 
                                                           worst_precision[0]*100,
                                                           worst_precision[1]))
        print(' Recall: {:.3f} ({:.1f}%) for {}'.format(worst_recall[0], 
                                                        worst_recall[0]*100,
                                                        worst_recall[1]))                 
    spent = time() - begin
    if verbose:
        print('process time:{:.4f} seconds'.format(spent))
        
    return (accuracy, precision, recall)


#########1#########2#########3#########4#########5#########6#########7#########8
def fn_scores_report(y_test, 
                     y_pred,
                     verbose=False):
    '''This function tests the model, giving a report for each label in y.
    It shows metrics, for reliable trained labels
    If the label could not be trained, it gives a warning about it 
    Inputs:
      - y_test (mandatory) - the y data for testing
      - y_pred (mandatory) - the y predicted for the model
      - verbose (optional) - if you want some verbosity (default=False)
    Output:
      - none
    '''
    print('ALERT: this function will be deprecated!')
    print('use fn_scores_report2 instead')
    print('###function scores_report started')
    begin = time()
    #index for column
    i = 0
    corrected_accuracy = False
    #consider_labels = []

    for column in y_test:
        print('######################################################')
        print('*{} -> label iloc[{}]'.format(column, i))
        
        #test for untrained column, if passes, shows report
        alfa = y_pred[:, i]
        if (pd.Series(alfa) == 0).all(): #all zeroes on predicted
            report = "  - as y_pred has only zeroes, report is not valid"
            #corrected_accuracy = True
        else:
            report = classification_report(y_test[column], 
                                           alfa)
            #consider_labels.append(i)
        print(report)
        i += 1
     
    #old accuracy formula (not real)
    accuracy = (y_pred == y_test.values).mean()
    
    #if corrected_accuracy:
    #    accuracy = f1_score(y_test, 
    #                        y_pred, 
    #                        average='weighted') 
    #                        #labels=consider_labels) #np.unique(y_pred))
    #else:
    #    accuracy = f1_score(y_test, 
    #                        y_pred, 
    #                        average='weighted')
    #                        #labels=consider_labels)

    print('Model Accuracy: {:.3f} ({:.1f}%)'.format(accuracy, accuracy*100))
    
    spent = time() - begin
    if verbose:
        print('process time:{:.4f} seconds'.format(spent))
        
    #return report

#########1#########2#########3#########4#########5#########6#########7#########8
def __main__():
  print('Second library for Udacity courses.')
  #12 useful functions in this package!
    
if __name__ == '__main__':
    main()
||||||| 53ccd8a
=======
#import matplotlib.patches as mpatches
#import matplotlib.patches as mpatches
#import matplotlib.style as mstyles
#import matplotlib.pyplot as mpyplots
#from matplotlib.figure import Figure
#import seaborn as sns

import re
import pandas as pd
from time import time
import numpy as np
import math #sorry, I need both!

#natural language toolkit
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

#metrics for Classifiers
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_cat_condenser(subset,
                     name='df',
                     opperation='all_sub',
                     verbose=False):
    '''this function...
    Inputs:
      - subset - 
      - name (optional) - the name of the dataset (default='df')
      - opperation (optional) - choose one opperation
        - 'all_subs' (default) - for taking all sub sets
      - verbose (optional) - if you want some verbosity (default=False)
    Outputs:
      - in a form of a Tuple, containing 3 items, as
      - Tuple[0] - the filtering statement, as a text string
      - Tuple[1] - the set name, as a text string
      - Tuple[2] - the list of subsets, as a Python list
    This function can be lately expanded for:
      - automatically selecting a Boolean crieteria for multi-filtering
      - including the "~" (not) statement just before the filter condition
      - allowing Integer entries for element, as 1
      - verifying consistencies (e.g., condition can be only "^" or "&"
    '''
    if verbose:
        print('###function cat_condenser started')
    
    begin = time()
        
    #paramameters for processing
    opperator='==' 
    super_item = 'related'
    #other_super = ['request', 'offer', 'direct_report']
    if opperation == 'empty_sub':
        element = '0' 
        condition = '&'
    else:
        element = '1' 
        condition = '^'    

    if subset == 'aid':
        set_item = 'aid_related'
        sub_lst = ['food', 'shelter', 'water', 'death', 'refugees', 'money', 
                   'security', 'military', 'clothing', 'tools', 'missing_people', 
                   'child_alone', 'search_and_rescue', 'medical_help', 
                   'medical_products', 'aid_centers', 'other_aid']
    elif subset == 'wtr':
        set_item = 'weather_related'
        sub_lst = ['earthquake', 'storm', 'floods', 'fire', 'cold', 
                   'other_weather']
    elif subset == 'ifr':
        set_item = 'infrastructure_related'
        sub_lst = ['buildings', 'transport', 'hospitals', 'electricity', 
                   'shops', 'other_infrastructure']
    else:
        raise Exception('invalid category for subset')
            
    out_str = fn_create_string(sub_lst=sub_lst,
                               dataset=name,
                               opperator=opperator,
                               element=element,
                               condition=condition,
                               verbose=verbose)        
        
    if opperation == 'all_sub':
        if verbose:
            print('processing for subset ended')
        output = out_str
    elif opperation == 'sub_not_main':
        if verbose:
            print('processing for sub_not_main ended')
        output = "(" + name + "['" + set_item + "'] == 0) & (" + out_str + ")"
    elif opperation == 'main_not_sub':
        if verbose:
            print('processing for main_not_sub ended')
        output = "(" + name + "['" + set_item + "'] == 1) & ~(" + out_str + ")"
    elif opperation == 'empty_sub':
        if verbose:
            print('processing for empty subset ended')
        output = "(" + name + "['" + set_item + "'] == 1) & (" + out_str + ")"
    else:
        raise Exception('category is invalid')

    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))

    return (output, set_item, sub_lst)

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_count_valids(dataset, 
                    field, 
                    criteria=1, 
                    verbose=False):
    '''This function...
    Inputs:
      - dataset
      - field
      - criteria
      - verbose -
    Output...
    '''
    if verbose:
        print('###counting function initiated')

    begin = time()  

    result = dataset[field][dataset[field] == criteria].sum()
        
    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
        
    return result

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_string(sub_lst,
                     dataset,
                     opperator,
                     element,
                     condition,
                     verbose):
    '''This function...
    Inputs:
      - set_item - 
      - sub_lst - 
      - verbose - 
    Output: filtering string for dataset opperations
    '''
    if verbose:
        print('###function create_string started')
    
    begin = time()
    string = ''

    for item in sub_lst:
        string = string + "(" + dataset + "['" + item + "'] " + opperator + \
" " + element + ")" + " " + condition + " "
        
    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))

    return string[:-3]

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_getKey(item):
    '''This is an elementary function for returning the key from an item 
    from a list
    Input:
      - an item from a list
    Output it´s key value
    It is necessary to run the ordered list function
    '''
    return item[0]

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_group_check(dataset,
                   subset,
                   correct=False,
                   shrink=False,
                   shorten=False,
                   verbose=False):
    '''This funtion...
    Inputs:
      - dataset (mandatory) - Pandas dataset
      - subset (mandatory) - text string
      - correct (optional) - if you want to correct inconsistencies
        (default=False)
      - shrink (optional) - if you want to shrink dataframe for the 
        focused dataset (default=False)
      - shorten (optional) - filter rows just for the main group
      - verbose (optional -
    Output:
      - corrected dataframe, for database theory about inconsistencies
    *future implementation, correct & shrink for other criteria than 1
    '''
    if verbose:
        print('###function group_check started')
    
    begin = time()
    
    #retrieving the name of the dataset, or using a default name
    #try:
    #    dataset.name
    #    name = dataset.name
    #except AttributeError:
    #    name = 'df'
    name = 'dataset'
    
    ###test for main class counting
    #I need this early calling, justo to get the main set name
    mainotsub = fn_cat_condenser(subset=subset,
                                 name=name,
                                 opperation='main_not_sub')
    main_class = mainotsub[1]
    count_main = dataset[dataset[main_class] == 1].shape[0]
    if verbose:
        print('  - count for main class:{}, {} entries'.\
format(main_class, count_main))
        
    ###test for main, without any sub-category 
    count_mainotsub = dataset[eval(mainotsub[0])].shape[0]
    
    if verbose:
        print('  - for main, without any sub-categories,  {} entries'.\
format(count_mainotsub))
        
    ###test for subcategories counting
    count_subs = dataset[eval(fn_cat_condenser(subset=subset,
                                               name=name,
                                               opperation='all_sub')[0])].shape[0]
    if verbose:
        print('  - for subcategories,  {} entries'.format(count_subs))

    ###test for sub, without main registered (data inconsistency)
    subnotmain = fn_cat_condenser(subset=subset,
                                  name=name,
                                  opperation='sub_not_main')
    count_subnotmain = dataset[eval(subnotmain[0])].shape[0]
    if verbose:
        print('  - for lost parent sub-categories,  {} entries'.\
format(count_subnotmain))
        
    if correct:
        #correcting to 1 - future: other criteria
        dataset.loc[dataset[eval(subnotmain[0])].index, subnotmain[1]] = 1
        #checking the correction
        subnotmain = fn_cat_condenser(subset=subset,
                                      name=name,
                                      opperation='sub_not_main')
        count_subnotmain = dataset[eval(subnotmain[0])].shape[0]
        if verbose:
            print('    *correcting, new count: {} entries'.\
format(count_subnotmain))
            
    if shrink:
        new_cols = ['message', 'genre', 'if_blank']
        new_cols.append(subnotmain[1]) #append the group column
        new_cols = new_cols + subnotmain[2]
        dataset = dataset[new_cols] #shrink for selected columns
        if verbose:
            print('    *shrinking, dataset now have: {} columns'.\
format(dataset.shape[1]))
            
    if shorten: #future:create other criteria
        dataset = dataset[dataset[subnotmain[1]] == 1]
        if verbose:
            print('    *shortening, dataset now have: {} lines'.\
format(dataset.shape[0]))
    
    end = time()
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))

    return dataset

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_labels_report(dataset, 
                     max_c=False, 
                     verbose=False):
    '''This function...
    Inputs:
      - dataset (mandatory) - the target dataset for reporting about
      - max_c (optional) - maximum counting - if you want to count for all elements,
        set it as False - (default=False)
    Output:
      - no output, only a report function!
    '''
    begin = time()
    
    expand_lst = ['related', 'request', 'offer', 'aid_related', 
                  'infrastructure_related', 'weather_related', 
                  'direct_report']
    
    aid_lst = ['food', 'shelter', 'water', 'death', 'refugees', 'money', 
               'security', 'military', 'clothing', 'tools', 'missing_people', 
               'child_alone', 'search_and_rescue', 'medical_help', 
               'medical_products', 'aid_centers', 'other_aid']
    
    weather_lst = ['earthquake', 'storm', 'floods', 'fire', 'cold', 
                   'other_weather']
    
    infrastructure_lst = ['buildings', 'transport', 'hospitals', 'electricity', 
                          'shops', 'other_infrastructure']
    
    expand_list = expand_lst + aid_lst + weather_lst + infrastructure_lst
    total = dataset.shape[0]
    counts = []

    #count for labels - not yet ordered!
    for field in expand_list:
        count = fn_count_valids(dataset=dataset, field=field)
        percent = 100. * (count / total)
        counts.append((count, field, percent))
        #if verbose:
        #    print('{}:{} ({:.1f}%)'.format(field, count, percent))
        
    #sort it as sorted tuples
    sorted_tuples = sorted(counts, key=fn_getKey, reverse=True)

    i=1
    c=2

    for cat in sorted_tuples:
        count, field, percent = cat
        print('{}-{}:{} ({:.1f}%)'.format(i, field, count, percent))
        if max_c:
            if c > max_c:
                break
        i += 1
        c += 1

    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
        
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_subcount_lists(column, 
                      verbose=False):
    '''This function takes a column that have a list, iterate them and 
    count unique items for each registry.
    The objective is to count different individuals that are nested.
    It also returns the sum for empty lists, it they exist.
    Inputs:
      - col (mandatory) - the column containing a list to be harshed - Pandas 
        Series
      - verbose (optional) - it is needed some verbosity, turn it on - 
        Boolean (default=False)
    Output:
      - a Dictionnary with the counting for each item, plus the number of rows 
        with NaN
    Obs: this is an adaptation from the fn_subcount_cols, from my authory, for 
         another Udacity project. The original version takes data from columns
         as string of characters, each element sepparated by ";". Now it takes
         lists, already sepparated.
    '''
    begin = time()
        
    #I already know that I want these entries, even if they finish as zero!    
    items_dic = {'empty_lists': 0,
                 'valid_rows': 0}
    harsh_dic = {} #temporary dictionnary for harshing
    
    for list in column:     
        if len(list) == 0: #ampty list (could be dangerous for our project!)
            if verbose:
                print('*empty list!')
            items_dic['empty_lists'] += 1
        else:
            #It may be necessary to remove all spaces inside the harshed item
            #I found the best way to do this at Stack Overflow, here:
            #https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string
            if verbose:
                print('splitted registry:', list)
            items_dic['valid_rows'] += 1
            
            for element in list:
                if verbose:
                    print('element for accounting:', element)
                if element in harsh_dic:
                    harsh_dic[element] += 1
                else:
                    harsh_dic[element] = 1

    #Why I made this strange sub-dictionnary insertion?
    #So, I think of a kind of Json structure will be really useful for my 
    #Udacity miniprojects (Yes, I am really motivated to study Json... it looks 
    #nice for my programming future!)
    items_dic['elements'] = harsh_dic
    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
        print('*************')
        print('dictionnary of counting items created:')
        print(items_dic)
    
    return items_dic

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_test(x, 
            verbose=False):
    '''This function...
    Inputs:
      - x
      - verbose
    Output:
    '''
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

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_tokenize(msg_text, 
                lemmatize=True, 
                rem_city=False,
                agg_words=False,
                rem_noise=False,
                elm_short=False,
                unhelpful_words=[],
                great_noisy=False,
                verbose=False):
    """This functions turns brute messages into tokens, for Machine Learning 
    training
    Inputs:
      - msg_text - string (mandatory) - a text string (not too long), as a 
        Tweeter message
      - lemmatize - boolean (optional) - if you want to run lemmatizer over 
        tokenizer, please
        turn it on (default=False)
      - rem_city - boolean (optional) - list of cities > than 100.000 
        inhabitants, to remove from
        messages (default=False) 
      - verbose - boolean (optional) - if you need some verbosity, turn it on 
        (default=False) 
    Output:
      - a list of tokens (reduced meaningful words)
    New addictions, ver 1.1:
      - built an aggregate function, to prevent duplicate words on a tokenized 
        string (as ['food', ... 'food', ...])
      - built a unnuseful words list, to remove common communication words, as 
        'thanks' and other noisy words for Machine Learning training
      - built an ellimination for too short words
    New Imputs:
      - agg_words - boolean (optional) - if you want to aggregate a list as a 
        set and turning back into a list (default=False)
      - rem_noise - boolean (optional) - if you want to remove the words from a 
        predefined list
        (default=False)
      - unhelpful_words - list (optional) - if you want to provide your own 
        noise_words list, ativate it
        (default = [])
      - elm_short = boolean/integer (optional) - if you want to elliminate 
        words shorter than a number please provide a number (e.g. 3)
        (default=False)
    """
    #if verbose:
    #    print('###Tokenizer function started')
        
    if rem_city:
        #print('*special list for city removal is loading')
        df_countries = pd.read_csv('all.csv')
        df_countries = df_countries['name'].apply(lambda x: x.lower())
        countries = df_countries.tolist()
        
    #add ver 1.1
    if rem_noise and (len(unhelpful_words) == 0):
        unhelpful_words = ['thank', 'thanks', 'god', 'fine', 'number', 'area', 
            'let', 'stop', 'know', 'going', 'thing', 'would', 'hello', 'say', 
            'neither', 'right', 'asap', 'near', 'want', 'also', 'like', 'since', 
            'grace', 'congratulate', 'situated', 'tell', 'almost', 'hyme', 
            'sainte', 'croix', 'ville', 'street', 'valley', 'section', 'carnaval',
            'rap', 'cry', 'location', 'ples', 'bless', 'entire', 'specially', 
            'sorry', 'saint', 'village', 'located', 'palace', 'might', 'given', 
            'santo', 'jesus', 'heart', 'sacred', 'please', 'named', 'janvier', 
            'avenue', 'tinante', 'cross', 'miracle', 'street', 'abroad', 'someone', 
            'country', 'rue']
        #if verbose:
        #    print('*{} added words on noise filter'.format(len(unhelpful_words)))
    
    #add ver 1.2
    if great_noisy:
        noisy_words = ['people', 'help', 'need', 'said', 'country', 'government', 
            'one', 'year', 'good', 'day', 'two', 'get', 'message', 'many', 'region', 
            'city', 'province', 'road', 'district', 'including', 'time', 'new', 
            'still', 'due', 'local', 'part', 'problem', 'may', 'take', 'come', 
            'effort', 'note', 'around', 'person', 'lot', 'already', 'situation', 
            'see', 'response', 'even', 'reported', 'caused', 'village', 'bit', 
            'made', 'way', 'across', 'west', 'never', 'southern', 'january', 
            'least', 'zone', 'small', 'next', 'little', 'four', 'must', 'non', 
            'used', 'five', 'wfp', 'however', 'com', 'set', 'every', 'think', 
            'item', 'yet', 'carrefour', 'asking', 'ask', 'site', 'line', 'put', 
            'unicef', 'got', 'east', 'june', 'got', 'ministry', 'http', 'information', 
            'area', 'find', 'affected', 'relief', 'well', 'million', 'give','state', 
            'send', 'team', 'three', 'make', 'week', 'santiago', 'service', 'official', 
            'month', 'united', 'nation', 'world', 'provide', 'report', 'much', 
            'thousand', 'call', 'level', 'prince', 'organization', 'agency', 
            'according', 'another', 'along', 'back', 'morning', 'news', 'town', 
            'centre', 'long', 'answer', 'management', 'main', 'crisis', 'delmas', 
            'tuesday', 'department', 'end', 'others', 'etc', 'among', 'general', 
            'july', 'six', 'past', 'eastern', 'told', 'haitian']
    
    #First step, lowering the case and taking words
    #lowering, you reduce variability
    low_text = msg_text.lower()
    #I need to take only valid words
    #a good Regex can good a nice job on finding and cleaning words
    #I created only a basic one (very common Regex filter) <- enhance it later!
    gex_text = re.sub(r'[^a-zA-Z]', ' ', low_text)
    first_step = len(gex_text)
        
    #Second step, tokenize & remove stop words
    #a tokenizer reduce words for their nearest more common synonym
    col_words = word_tokenize(gex_text)
    #stop words are these that don´t have an intrinsic meaning
    #languages use them for better gramatical construction on phrases
    unnuseful = stopwords.words("english")
    output = [word for word in col_words if word not in unnuseful]
    second_step = len(output)

    #Optional Step, remove cities names form the text
    if rem_city:
        #if verbose:
        #    print('*optional city names removal started')
        output = [word for word in output if word not in countries]
        optional_step = len(output)
    
    #Optional Step, included on ver 1.1
    if rem_noise:
        output = [word for word in output if word not in unhelpful_words]
        optional_step2 = len(output)
    
    #Third step, lemmatize
    #a lemmatizer reduce words for their root form - reduce variability
    #normally they apply both, tokenizer and lemmatizer
    #they area a bit redundant, so we can disable lemmatizer
    if lemmatize:
        output = [WordNetLemmatizer().lemmatize(word) for word in output]
        third_step = len(output)
    
    #add ver 1.1
    if agg_words:
        output = list(set(output))
        agg_step = len(output)
    
    #add ver 1.1
    if elm_short:
        if isinstance(elm_short, int): #test if integer
            output = [word for word in output if len(word) >= elm_short]
            optional_step3 = len(output)
    
    #add ver 1.2
    if great_noisy:
        output = [word for word in output if word not in noisy_words]
        optional_step4 = len(output)
         
    spent = time() - begin
    if verbose:
        print('process time:{:.4f} seconds'.format(spent))
        if rem_city and rem_noise:
            print('Tokens-start:{}, token/stop:{}, remove cities:{} &noise:{}'.\
format(first_step, second_step, optional_step, optional_step2))
        elif rem_city:
            print('Tokens-start:{}, token/stop:{}, remove cities:{}'.\
format(first_step, second_step, optional_step))
        elif rem_noise:
            print('Tokens-start:{}, token/stop:{}, remove noise:{}'.\
format(first_step, second_step, optional_step2))
        else:
            print('Tokens-start:{}, token/stop:{}'.format(first_step, 
                                                          second_step))
        if lemmatize:
            print(' +lemmatizer:{}'.format(third_step))
        if elm_short:
            print(' +eliminate short:{}'.format(optional_step3))
            
        if great_noisy:
            print(' +eliminate noisy from 300:{}'.format(optional_step4))
    
    return output

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_tokenize_fast(msg_text, 
                     verbose=False):
    """This is the fast version for word tokenizer. It makes only one loop for 
    all the selected as best functions
    Inputs:
      - msg_text - string (mandatory) - a text string (not too long), as a 
        Tweeter message
      - verbose - boolean (optional) - if you need some verbosity, turn it on 
        (default=False) 
    Output:
      - a list of tokens (reduced meaningful words)
    """
    #if verbose:
    #    print('###Tokenizer function started')
        
    cleared = []
    unnuseful = stopwords.words("english")
    #marked for remove
    unhelpful_words = ['thank', 'thanks', 'god', 'fine', 'number', 'area', 'let', 
        'stop', 'know', 'going', 'thing', 'would', 'hello', 'say', 'neither', 
        'right', 'asap', 'near',  'also', 'like', 'since', 'grace', 'congratulate', 
        'situated', 'ville', 'street', 'valley', 'section', 'rap',  'location', 
        'ples', 'bless', 'entire', 'specially', 'sorry', 'saint', 'village', 
        'located', 'palace', 'might', 'given', 'santo', 'jesus', 'heart', 'sacred', 
        'named', 'janvier', 'avenue', 'tinante', 'cross', 'street', 'abroad', 
        'someone', 'country', 'rue', 'people',  'said', 'country', 'one', 'year', 
        'good', 'day', 'two', 'get', 'message', 'many', 'region', 'city', 'province', 
        'including', 'time', 'new',  'due', 'local', 'part',  'may', 'take', 'come', 
        'note', 'around', 'person', 'lot', 'already',  'see', 'response', 'even', 
        'village', 'bit', 'made', 'way', 'across', 'west', 'never', 'southern', 
        'january', 'least', 'zone', 'small', 'next', 'little', 'four', 'must', 'non', 
        'used', 'five', 'wfp', 'however', 'com', 'set', 'every', 'think', 'item', 
        'yet', 'site', 'line', 'put', 'got', 'east', 'june', 'got', 'ministry', 'http',  
        'area', 'well', 'state', 'send', 'three', 'make', 'week', 'service', 'told',
        'official', 'world', 'much', 'level', 'prince', 'road', 'district', 'main',
        'according', 'another', 'along', 'back',  'town', 'centre', 'long', 'management', 
        'tuesday', 'department', 'end', 'others', 'etc', 'among', 'general', 'july', 
        'imcomprehensibley', 'incomprehensible', 'six', 'past', 'eastern', 'could',
         'previous', 'regards', 'cul', 'pitrea', 'northern']
    #not removing (I´m in doubt about them!)
    in_doubt = ['carrefour', 'delmas', 'cuba', 'haitian', 'haiti','affected', 'relief',
        'problem', 'united', 'nation', 'team', 'provide', 'report', 'million', 'give',
        'santiago', 'month', 'morning', 'news', 'help', 'need', 'cry', 'please', 'still',
        'crisis', 'answer', 'reported', 'caused', 'asking', 'ask', 'thousand', 'information',
        'want', 'call', 'effort', 'situation', 'tell', 'almost', 'hyme', 'sainte', 'croix',
        'miracle', 'unicef', 'find', 'organization', 'agency', 'carnaval', 'government']
    
    #if you want to try both lists for removal, please uncomment the following line
    #unhelpful_words = unhelpful_words + in_doubt     
    
    #lowering, you reduce variability
    low_text = msg_text.lower()

    #take only valid words by Regex
    gex_text = re.sub(r'[^a-zA-Z]', ' ', low_text)
        
    #tokenize & remove stop words
    col_words = word_tokenize(gex_text)
        
    #remove stop words + unhelpful words + lemmatize
    for word in col_words:
        if ((word not in unnuseful) and (word not in unhelpful_words)) and (len(word) >= 3):
            WordNetLemmatizer().lemmatize(word)
            cleared.append(word)
            
        cleared = list(set(cleared))
        
    if verbose:
        print(cleared)
             
    return cleared

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_valids_report(dataset, 
                     verbose=False):
    '''This function...
    Input:
      - dataset
    Output:
      - none
    '''
    print('###function valids_report started')
    begin = time()
    total = dataset.shape[0]
    field = 'if_blank'
    count = fn_count_valids(dataset=dataset, 
                            field=field, 
                            criteria=True)
    percent = 100. * (count / total)
    print('  *{}:{} ({:.1f}%)'.format(field, count, percent))
    
    spent = time() - begin
    if verbose:
        print('process time:{:.4f} seconds'.format(spent))
                
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_scores_report(y_test, 
                     y_pred,
                     verbose=False):
    '''This function tests the model, giving a report for each label in y
    Inputs:
      - y_test - 
      - y_pred -
    Output:
      - none
    '''
    print('###function scores_report started')
    start = time()
    #index for column
    i = 0
    corrected_accuracy = False
    #consider_labels = []

    for column in y_test:
        print('######################################################')
        print('*{} -> label iloc[{}]'.format(column, i))
        
        #test for untrained column, if passes, shows report
        alfa = y_pred[:, i]
        if (pd.Series(alfa) == 0).all(): #all zeroes on predicted
            report = "  - as y_pred has only zeroes, report is not valid"
            #corrected_accuracy = True
        else:
            report = classification_report(y_test[column], 
                                           alfa)
            #consider_labels.append(i)
        print(report)
        i += 1
     
    #old accuracy formula (not real)
    accuracy = (y_pred == y_test.values).mean()
    
    #if corrected_accuracy:
    #    accuracy = f1_score(y_test, 
    #                        y_pred, 
    #                        average='weighted') 
    #                        #labels=consider_labels) #np.unique(y_pred))
    #else:
    #    accuracy = f1_score(y_test, 
    #                        y_pred, 
    #                        average='weighted')
    #                        #labels=consider_labels)

    print('Model Accuracy: {:.3f} ({:.1f}%)'.format(accuracy, accuracy*100))
    
    spent = time() - begin
    if verbose:
        print('process time:{:.4f} seconds'.format(spent))
        
    #return report

#########1#########2#########3#########4#########5#########6#########7#########8
def __main__():
  print('Second library for Udacity courses.')
  #12 useful functions in this package!
    
if __name__ == '__main__':
    main()
>>>>>>> c9dcb35a089471818e15b0f115fec0b40700eb4f
