# SecondProject

## Second Udacity Project (unfinished)

This project is about building a **Machine Learning Pipeline**. The objective of this Pipeline is to automatically classify short messages (as Tweeter), in 36 different predefined categories (labels) for **Disaster Report**. This is the second part of **Udacity Data Science** nanodegree.

---

#### Modules description

The project is divided in two main parts. And the files involded are listed below.

`udacourse2.py` → python 3 code, as a library of useful functions for both parts

The **first** one, is an ETL (Extract, Transform, Load) for taking a .csv file, rearrange the data in classes and save the data corrected data into SQLite format, as a Database Table named Messages.

`messages.csv` → file containing the raw data to be processed

`process_data.py` → Python 3 code, with whe ETL functions

`ETL Pipeline Preparation.ipynb` → Jupyter Notebook containing all the steps for building the ETL

`ETL Pipeline Testing.ipynb` → Jupyter Notebook for calling and testing `process_data.py` functions

The **second** part is a Machine Learning Pipeline for reading the data on SQLlite format, preprocess it, tokenize, prepare the data and train a Machine Learning Classifier. 

`Messages.db` → file containing the preprocessed data for Machine Learning training

`train_classifier.py` → Python 3 code, with the Train Classifief functions

`ML Pipeline Preparation.ipynb` → Jupyter Notebook documenting all the steps for building the Machine Learning Pipeline

`ML Pipeline Condensing.ipynb` → Jupyter Notebook for condensation of all the steps from `ML Pipeline Preparation.ipynb`, in order to turn it into useful functions

`ML Pipeline Testing.ipynb` → Jupyter Notebook for calling and testing `train_classifier.py` functions

---

Project started at 05/2021

Versions:

- 0.1 Alfa - 1.2 update: small model improvement

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
- 1.16 update: new fn_scores_report2 function created

- 1.17 update: metrics changed, so my choice may change too!
- 1.17b update: for Naïve Bayes updated new, more realistic metrics based on 10-top labels
- 1.17c update: for Linear Support Vector updated new, more realistic metrics based on 10-top labels
- 1.17d update: for k-Nearest updated new, more realistic metrics based on 10-top labels

- 1.18 update: letting the tokenizer take the same word more than once:
- 1.18b update: for Naïve Bayes letting the tokenizer take the same word more than once
- 1.18c update: for Linear Support Vector Machine letting the tokenizer take the same word more than once
- 1.18d update: now my Classifier was changed to Linear SVC. The explanations for my choice rests above

- 1.19 update: not removing anymore any column and preserving 36 columns at the Labels dataset
 
##### This project is under MIT Licence
 
"A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code."

Permissions:

- Commercial use
- Modification
- Distribution
- Private use

Limitations:
- Liability
- Warranty
