import json
import plotly
#import plotly.express as px
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def fn_labels_report(dataset, 
                     max_c=False,
                     data_ret=False,
                     label_filter=False,
                     verbose=False):
    '''This is a report only function!
    Inputs:
      - dataset (mandatory) - the target dataset for reporting about
      - max_c (optional) - maximum counting - if you want to count for all 
        elements, set it as False (default=False)
      - data_ret (optional) - if you want to return a Pandas Dataframe with
        the results (default=False)
      - verbose (optional) - if you want some verbosity (default=False)
    Output:
      - no output, shows reports about the labels counting
    '''
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
    if not label_filter: #all the labels
        expand_list = expand_lst + aid_lst + weather_lst + infrastructure_lst
    elif label_filter == 'main':
        expand_list = ['related', 'request', 'offer', 'direct_report']
    elif label_filter == 'related':
        expand_list = ['aid_related', 'infrastructure_related', 'weather_related']
    elif label_filter == 'expand':
        expand_list = expand_lst
    elif label_filter == 'aid':
        expand_list = aid_lst
    elif label_filter == 'weather':
        expand_list = weather_lst
    elif label_filter == "infra":
        expand_list = infrastructure_lst
    else:
        raise Exception('invalid label_list parameter')
    total = dataset.shape[0]
    counts = []
    #count for labels - not yet ordered!
    for field in expand_list:
        count = fn_count_valids(dataset=dataset, field=field)
        percent = 100. * (count / total)
        counts.append((count, field, percent))
    #sort it as sorted tuples
    sorted_tuples = sorted(counts, key=fn_getKey, reverse=True)
    i=1
    c=2
    tuples_lst=[]
    for cat in sorted_tuples:
        count, field, percent = cat
        print('{}-{}:{} ({:.1f}%)'.format(i, field, count, percent))
        tuples_lst.append((field, count, percent))
        if max_c:
            if c > max_c:
                break
        i += 1
        c += 1
    df_report = pd.DataFrame(tuples_lst, columns = ['label', 'count', 'percentage'])
    if data_ret:
        return df_report

def fn_getKey(item):
    '''This is an elementary function for returning the key from an item 
    from a list
    Input:
      - an item from a list
    Output it´s key value
    It is necessary to run the ordered list function
    '''
    return item[0]
    
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
    result = dataset[field][dataset[field] == criteria].sum()
    return result
    
# load data
engine = create_engine('sqlite:///../data/Messages.db')
df = pd.read_sql_table('Messages', engine)
# load model
model = joblib.load("../models/classifier.pkl")
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    tuples_main = fn_labels_report(dataset=df,
                                   label_filter='main',
                                   data_ret=True,
                                   max_c=False)
    tuples_related = fn_labels_report(dataset=df,
                                      label_filter='related',
                                      data_ret=True,
                                      max_c=False)
    tuples_related1 = fn_labels_report(dataset=df,
                                       label_filter='aid',
                                       data_ret=True,
                                       max_c=False)
    tuples_related2 = fn_labels_report(dataset=df,
                                       label_filter='weather',
                                       data_ret=True,
                                       max_c=False)
    tuples_related3 = fn_labels_report(dataset=df,
                                       label_filter='infra',
                                       data_ret=True,
                                       max_c=False)
    # create visuals
    graphs = [{
        'data': [{'type': 'pie',
                  'labels': tuples_main['label'],
                  'values': tuples_main['percentage'],
                 }],
        'layout': {'title': {'text': 'Main Categories - relative percentages'}}
    },
    
             {
        'data': [Bar(x=tuples_main['label'],
                     y=tuples_main['percentage'],
                     marker=dict(color="Magenta")),
                ],
        'layout': {'title': 'Main Categories - total percentages',
                   'yaxis': {'title': "Count"},
                   'xaxis': {'title': ""},
                  },
    },
             {
        'data': [Bar(x=tuples_related['label'],
                     y=tuples_related['percentage'],
                     marker=dict(color="Cyan")),
                ],
        'layout': {'title': 'Related Category - total percentages',
                   'yaxis': {'title': "Count"},
                   'xaxis': {'title': ""},
                  },
    },
             {
        'data': [Bar(x=tuples_related1['label'],
                     y=tuples_related1['percentage'],
                     marker=dict(color="Green")),
                ],
        'layout': {'title': 'Aid Related Subategory - total percentages',
                   'yaxis': {'title': "Count"},
                   'xaxis': {'title': ""},
                   'tickangle': -35
                  },
    },        
             {
        'data': [Bar(x=tuples_related2['label'],
                     y=tuples_related2['percentage'],
                     marker=dict(color="Green")),
                ],
        'layout': {'title': 'Weather Related Subategory - total percentages',
                   'yaxis': {'title': "Count"},
                   'xaxis': {'title': ""},
                   'tickangle': -35
                  },
    },        
             {
        'data': [Bar(x=tuples_related3['label'],
                     y=tuples_related3['percentage'],
                     marker=dict(color="Green")),
                ],
        'layout': {'title': 'Infrastructure Related Subategory - total percentages',
                   'yaxis': {'title': "Count"},
                   'xaxis': {'title': ""},
                   'tickangle': -35
                  },
    },        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()