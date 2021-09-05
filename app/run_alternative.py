import numpy as np
from collections import Counter
from pprint import pprint

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('clean_data', engine)

    # extract data needed for visuals
    # calculate percentages rounded to two decimal places
    genre_per = round(100*genre_counts/genre_counts.sum(), 2)
    
    category_related_counts = df.groupby('related').count()['message']
    category_related_names = ['Related' if i==1 else 'Not Related' for i in list(category_related_counts.index)]

    requests_counts = df.groupby(['related','request']).count().loc[1,'message']
    category_requests_names = ['Requests' if i==1 else 'Not Requests' for i in list(requests_counts.index)]
    
    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values
    # Top five categories count
    top_category_count = df.iloc[:,4:].sum().sort_values(ascending=False)[1:11]
    top_category_names = list(top_category_count.index)

    # word cloud data
    #message_list = df['message'].unique().tolist()
    #messagelen_list = [len(tokenize(message)) for message in message_list]
    repeated_words=[]            # contain all repated words
                                                          
    for text in df['message'].values:
        tokenized_ = tokenize(text)
        repeated_words.extend(tokenized_)

    word_count_dict = Counter(repeated_words)      # dictionary having words counts for all words\
                                                        
    
    sorted_word_count_dict = dict(sorted(word_count_dict.items(),
                                          key=lambda item:item[1], reverse=True))
                                          # sort dictionary by\
                                                          # values
    topwords, topwords_20 =0, {}

    for k,v in sorted_word_count_dict.items():
        topwords_20[k]=v
        topwords+=1
        if topwords==20:
            break
    words=list(topwords_20.keys())
    pprint(words)
    count_props=100*np.array(list(topwords_20.values()))/df.shape[0]
  
    # create visuals
    graphs = [
        {
            "data": [
              {
                "type": "pie",
                "hole": 0.6,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": genre_per,
                  "y": genre_names
                },
                "marker": {
                  "colors": [
                    "Red",
                    "Blue",
                    "Yellow"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre_names,
                "values": genre_counts
              }
            ],
            "layout": {
              "title": "Distribution of Messages by Genre"
            }
        },
       
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean,
                    #orientation = 'h',
                    marker=dict(color="Magenta")
                    
                )
                  
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "",
                    'tickangle': -35
                    #'rotation':90
                }
            }
            
        },
         
        {
            'data': [
                Bar(
                    x=category_requests_names,
                    y=requests_counts,
                    marker=dict(color="")
                                    )
            ],

            'layout': {
                'title': 'Distribution of Request Messages <br> out of all Disaster Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        }
    ]