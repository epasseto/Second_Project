# Second Project

## Disaster Response Pipeline

This project is about building a **Machine Learning Pipeline**. The objective of this Pipeline is to automatically classify short messages (as Tweeter), in 36 different predefined categories (labels) for **Disaster Report**. This is the second part of **Udacity Data Science** nanodegree.

---

#### Modules description

The project is divided in two main parts. And the files involded are listed below.

`udacourse2.py` → Python 3 code, as a library of useful functions for both parts of the project

### The **first** part

Is an **ETL** (Extract, Transform, Load) for taking a .csv file, rearrange the data in classes and save the data corrected data into SQLite format, as a Database Table named Messages.

`disaster_messages.csv` → file containing the raw data to be processed (input: later it will be converted in our X-data, messages for training) 

`disaster_categories.csv`→ file containing the raw data for labels to be processed (input: later it will be converted into our y-categories, labels for classification)

`process_data.py` → full functional Python 3 code, containing main() caller and the ETL functions

`Messages.db` → SQLite file containing the processed data for feeding the Machine Learning Classifier (output)

#### additional files

`ETL Pipeline Preparation.ipynb` → Jupyter Notebook containing all the steps for building the ETL (documentation for the steps made to create this first ETL)

`ETL Pipeline Testing.ipynb` → Jupyter Notebook for calling and testing `process_data.py` functions

### additional data sources

`messages.csv` → file containing the original raw data to be processed (input: later it will be converted in our X-data, messages for training) 

`categories.csv`→ file containing the original raw data for labels to be processed (input: later it will be converted into our y-categories, labels for classification)

### The **second** part

Is a **Machine Learning Pipeline** for reading the data on SQLlite format, preprocess it, tokenize, prepare the data and train a Machine Learning Classifier. 

`Messages.db` → SQLite file containing the preprocessed data for Machine Learning training (input)

`train_classifier.py` → full functional Python 3 code, with the Train Classifief functions

#### additional files 

`ML Pipeline Preparation.ipynb` → Jupyter Notebook documenting all the steps for building the Machine Learning Pipeline

`ML Pipeline Condensing.ipynb` → Jupyter Notebook for condensation of all the steps from `ML Pipeline Preparation.ipynb`, in order to turn it into useful functions

`ML Pipeline Testing.ipynb` → Jupyter Notebook for calling and testing `train_classifier.py` functions

#### Flask \ frontend files

`run.py` → Python 3 file for running the app. A first version of this file was offered by Udacity, and was customized here. It is needed to run the file, after making the data preparation and training the Classificer with the SQLite table data, and then run this file as `python run.py`, then starting the environment with `env|grep WORK`, taking the correct address for seeing the app working

`go.html` → HLML file, provided by Udacity

`master.html` → HTML file, provide by Udacity

---

### How to use:

The main files are `process_data.py` and `train_classifier.py`. The idea is to run both of the files using a Python terminal (you can use Anaconda Terminal to run them), giving the instructions to process the data, save it in a SQLite database, read it, train a Machine Learning Classifier and savig it as a Picke file. 

The original projec have the following structure (you can find it at Udacity Data Science Course at the [link](https://classroom.udacity.com/nanodegrees/nd025/parts/ba5d2f25-63d2-4db8-afb1-7a37dd792b4a/modules/1112326c-bdb1-4119-b907-4098a0e4277d/lessons/743ff0a6-7500-4de6-8477-ea822eeda8b8/concepts/6f0d69e6-1f5e-413e-8176-6b80a9bc8ad3) - perhaps you need autorization to access this area

>- app
>>- template
>>>- master.html  ← main page of web app
>>>- go.html  ← classification result page of web app
>>- run.py  ← Flask file that runs app
>- data
>>- disaster_categories.csv  ← data to process (equivalent to categories.csv)
>>- disaster_messages.csv  ← data to process (equivalent to messages.csv)
>>- process_data.py (our ETL pipeline)
>>- udacourse2.py (supporting library)
>>- Disaster.db   ← database to save clean data to (equivalent to Messages.db)
>- models
>>- train_classifier.py (our Classifier pipeline) 
>>- udacourse2.py (supporting library)
>>- classifier.pkl  ← saved model (equivalent to model.pkl)

- README.md (our README file)

### How it works:

**First**, you need to run `process_data.py` with the correct parameters in a Terminal:

>- it will take `disaster_categories.csv` and `disaster_messages.csv`(the data on these files can be changed later, but you need to keep the same structure!)
>- it will process it and save in a table in `InsertDatabaseName.db` for later use

Call example for Udacity environment:

- on /home/workspace/data

- call `python process_data.py disaster_messages.csv disaster_categories.csv sqlite:///Messages.db` ← `-v` for verbosity on processing

**Second**, you will ned to run `train_classifier.py` with the correct parameters in a Terminal:

>- it will take the saved data in `InsertDatabaseName.db`
>- it will create a Machine Learning Pipeline for it, train it an save it on a Picle file `classifier.pkl` 

Call example for Udacity environment:

- on /home/worskpace/models

- call `python train_classifier.py sqlite:////home/workspace/data/Messages.db classifier.pkl -v -g` ← `-g` for perform Grid Search, `-v` for verbosity on processing

*Observe that the names of these files are only **illustrative**, as you can change them using your function call*

Later a Flask application will take the trained Classifier and will open a HTML window. In this window you can insert a new Teeeter-type message (short text message) and the system will try to classify it in the pre-trained labels.

### How to call the main functions

There are some default values for each function. So if you call

`>>>python process_data.py` → it will be equivalent to call `>>python process_data.py messages.csv categories.csv sqlite:///Messages.db`

These three basic parameters are:

>- messages.csv - filepath for your **messages** training data file
>- categories.csv - filepath for your **categories** labels file
>- ...Messages.db - filepath for your **SQLite** database file

There are some **extra** parameters for `process_data.py`. For using them you will need to make a full-calling (the name of the function + the path for the extra functions calling). You can call more than one of them. Te full documentation about what they do is inside the funcions on the library. The basic parameters are:

`-a` → add_report - if you want adittional later report (default=False)

`-c` → categories_index - if you want to alter (default='id')

`-v` → verbose - if you want verbosity (default=False)

`-m` → messages_index - if you want to alter (default='id')

`-s` → cat_sepparation - if you want to alter (default=';')

Examples of use:

`>>>python process_data msgs.csv resp.db -m=ID -c=index -s=; -v -a`

`>>>python process_data.py messages.csv disaster_categories.csv DisasterResponse.db -v`


`>>>python train_classifier.py` → it will be equivalent to call `>>>python train_classiifer.py sqlite:///Messages.db classifier.pkl`

There are some **extra** parameters for `train_classifier.py` too. For using them you will need to make a full-calling (the name of the function + the path for the 

`-a` → run metrics over ALL labels (not recommended, default=False) - run metris over the 10 main labels only

`-c` → C parameter for your Classificer (default=2.0)

`-e` → NOT remove duplicates (condense) tokens on a document. Sometimes it turns easier for the Classifier to fit best parameters, others not (default=True - remove duplicates) - not working temporally in this project due to an issue, waiting for be reinserted

`-g` → perform Grid Search over Adaboost before training for the best parameters. Please use it wisely, as it costs a lot of processing time!

`-l` → learning rate for Adaboost Classifier. It have a tradeoff with n_estimators, so consider to tune both parameters (default=0.5)

`-n` → number of maximum estimators for Adaboost (default=80)

`-p` → pre_tokenize - keep preprocessing tokenization column, for saving processing time. Obsservation: keeping this column turns the system faster, but may cause instability on Classifier training on Flask due to "pipeline leakage" (not recomended) (default=False) 

`-r` → remove columns - if you want to remove (un)trainable columns from your y-labels dataset (default=False)

`-s` → change Classifier from Adaboost (tree-type) to LSVM (support vector machine-type)

`-t` → test size for splitting your data (default=0.25)

`-v` → verbose - if you want some verbosity during the running (default=False)

Examples of use:

`>>>python train_classifier data.db other.pkl -C=0.5 -t=0.2 -r -v`

`>>>python train_classiifer.py sqlite:///Messages.db classifier.pkl -v`

---

Project started at 05/2021

Versions:

- 0.1..0.9 Alfa: incomplete releases

- 1.0 update: ETL Pipeline now is complete!

- 1.1 update: new parameters for ETL, Pipeline now can run under Terminal calling

- 1.2 update: small ETL model improvement

- 1.3 update: pre-tokenizer (a premature tokenization strategy) created, for removing untrainable rows

- 1.4 update: could absorb pre-tokenized column as a input for Machine Learning Classifier, saving time!

- 1.5 update - added hierarchical structure on labels, for checking and correcting unfilled classes that already have at least one subclass alredy filled
 
- 1.6 update: removed `related` column from the Labels dataset 

- 1.7 update: removed from training columns that contains **only zeroes** on labels

- 1.8 update: now I am using random_state parameter, so I can compare exactly the same thing, when using randomized processes, for ensuring the same results for each function call
 
- 1.10 update: prepared other Machine Learning Classifiers for training the data

- 1.11 update: preparation of k-Neighbors Classifier for training

- 1.12 update: preparation of a completely different kind of Machine Learning Classifier (Support Vector Machine Family)

- 1.13 update: implemented Grid Search for some sellected Classifiers

- 1.14 update: filtering for valid ones over critical labels

- 1.15 update: improved my customized function for other metrics

- 1.16 improvement: trying to create new fancy metrics for scoring my Classifiers


- 1.16b update: new fn_scores_report2 function created

- 1.17 update: metrics changed, so my choice may change too!

- 1.17b update: for Naïve Bayes updated new, more realistic metrics based on 10-top labels

- 1.17c update: for Linear Support Vector updated new, more realistic metrics based on 10-top labels

- 1.17d update: for k-Nearest updated new, more realistic metrics based on 10-top labels

- 1.18 update: letting the tokenizer take the same word more than once:

- 1.18b update: for Naïve Bayes letting the tokenizer take the same word more than once
- 1.18c update: for Linear Support Vector Machine letting the tokenizer take the same word more than once
- 1.18d update: now my Classifier was changed to Linear SVC. The explanations for my choice rests above

- 1.19 update: not removing anymore any column and preserving 36 columns at the Labels dataset

- 1.20a update: after running GridSearch on Adaboost
- 1.20b update: after running GridSearch on Adaboost, I could make some parameters testing

- 1.21 update: for preventing pipeline leakage using Picke, I modified train_data for having pre_tokenization preprocessing as optional

- 1.22 update: `condensing` bug correction. This attribute was causing some instability in the system. In a new future it will be active again. It is not fundamental for running the project

- 1.23 update `condensing` attribute was removed, due to an insuperable issue

- 1.24 update `condensing` attribute was re-inserted, but is still not functional

- 1.25 update hard coding was removed for `process_data` and `train_classifier` files

- 1.26 update `train_classifier` had a defective exception raising on main function call, `process_data` had a defective parameter name on main function. Corrected.
 
##### This project is under MIT Licence
 
"A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code."

Summary of MIT Licence (see on MIT Website for details)

**Permissions**:
- Commercial use
- Modification
- Distribution
- Private use

**Limitations**:
- Liability
- Warranty
