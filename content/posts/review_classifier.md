---
title: Imbalanced Product Review Sentiment Classification
date: 2024-02-29
author: Chandler Underwood
description: Using NLP, resampling, and surpervised machine learning techniques, I create a model to effectively classify the minority negative sentiment contained in the crocs review dataset.
ShowToc: true
TocOpen: true
---

## Class Imbalance
Class imbalance is a common problem when building classifiers in the machine learning world, and our awesome croc reviews data is unfortunately not so awesome from a class balance standpoint. Soon, we'll assign binary class labels based on the rating a customer gave with their review where we'll consider ratings of 2 stars or less to be negative sentiment and the remaining reviews as positive sentiment. As you'll see in a moment, the vast majority of reviews belong to the positive sentiment class, and I think that's great! I'm super happy so many people are enjoying their crocs as much as Grandma Louise. But...the shoe game is a competitive one, and if Crocs are going to remain at the top, they must continue to work on their flaws and improve. Therefore, the negative reviews are much more important than the positive ones. We will take steps in our model training to address the class imbalance issue, but the goal will remain the same throughout - build a classifier that can find the most important samples (the ones with negative sentiment) effectively.

### Quick Note on Positive / Negative Lingo
Traditionally, the positive class in a binary labeled dataset is the minority class of most interest / importance, and that will hold true in this project. Apologies for any confusion, but going forward when I reference the positive class, I will be referencing the set of negative sentiment reviews.

* Positive Class (1) - Negative Sentiment Review :(
* Negative Class (0) - Positive Sentiment Review :)

Below is a look at the class imbalance of our data.


```python
import pandas as pd
from collections import Counter
import seaborn as sns

df = pd.read_csv('/content/drive/MyDrive/croc_reviews.csv')
df['label'] = [0 if each > 2 else 1 for each in df['rating']]
sns.countplot(x = df['label'], hue = df['label'])
print("Only", "{:.0%}".format(label_count[1] / (label_count[1] + label_count[0])), "of our data belongs to the positive class.")
```
Only 6% of our data belongs to the positive class.
![png](/class_balance.png)
    


## Data Clean Up
Before we start addressing the class imbalance issue, let's clean up the reviews using the text cleaning function we've used before.


```python
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def clean_text(text):
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

df['clean_review'] = [clean_text(text_to_clean) for text_to_clean in df['review']]
df
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>date</th>
      <th>rating</th>
      <th>label</th>
      <th>clean_review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>!!!!!! E X C E L L E N T!!!!!!!!!!</td>
      <td>April 7, 2022</td>
      <td>5.0</td>
      <td>0</td>
      <td>e x c e l l e n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"They're crocs; people know what crocs are."</td>
      <td>April 3, 2021</td>
      <td>5.0</td>
      <td>0</td>
      <td>theyr croc peopl know croc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>- Quick delivery and the product arrived when ...</td>
      <td>March 19, 2023</td>
      <td>5.0</td>
      <td>0</td>
      <td>quick deliveri product arriv compani said woul...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>...amazing "new" color!! who knew?? love - lov...</td>
      <td>July 17, 2022</td>
      <td>5.0</td>
      <td>0</td>
      <td>amaz new color knew love love love</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0 complaints from me; this is the 8th pair of ...</td>
      <td>June 4, 2021</td>
      <td>5.0</td>
      <td>0</td>
      <td>complaint th pair croc ive bought like two mon...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9233</th>
      <td>I will definitely be buying again in many colo...</td>
      <td>August 25, 2021</td>
      <td>4.0</td>
      <td>0</td>
      <td>definit buy mani color reason materi feel thin...</td>
    </tr>
    <tr>
      <th>9234</th>
      <td>I wish I would have bought crocs a long time ago.</td>
      <td>April 8, 2021</td>
      <td>5.0</td>
      <td>0</td>
      <td>wish would bought croc long time ago</td>
    </tr>
    <tr>
      <th>9235</th>
      <td>wonderful. Gorgeous blue; prettier in person!</td>
      <td>April 27, 2022</td>
      <td>5.0</td>
      <td>0</td>
      <td>wonder gorgeou blue prettier person</td>
    </tr>
    <tr>
      <th>9236</th>
      <td>Wonerful. Very comfy, and there are no blister...</td>
      <td>April 8, 2021</td>
      <td>5.0</td>
      <td>0</td>
      <td>woner comfi blister feet unlik brand one</td>
    </tr>
    <tr>
      <th>9237</th>
      <td>Work from home - high arch need good support a...</td>
      <td>May 22, 2023</td>
      <td>5.0</td>
      <td>0</td>
      <td>work home high arch need good support comfort ...</td>
    </tr>
  </tbody>
</table>
<p>9238 rows × 5 columns</p>



## Strategy: Performance Metrics and Dealing with Imbalanced Data
Because we're dealing with imbalanced data and are most concerned with identifying the minority / positive class, we will focus on improving the recall score of our models on the test set. We will also watch F2 score, a modifed version of F1 score that increases the importance of recall in its calculation. Why F2 score? We are concerned with maximizing recall, but a model that predicts the minority class 100% of the time would achieve a perfect recall score. That doesn't help us very much, and F2 score will give us an understanding of how well the model can *differentiate* between the two classes along with how well the model can identify positive samples. Below are the formulas to calculate the performance metrics.

- Recall - TP / (TP + FN)
- Precision - TP / (TP + FP)
- F2 Score - (5 * Precision * Recall) / (4 * Precision + Recall)


There are a lot of ways to address the imbalanced data problem when training a classifier. In this project we're going to adopt the following strategy:
1. Implement multiple methods for resampling the data
2. Train multiple baseline models using each of the resampling methods to see which resampling / model combo performs the best out of the box based on F2 and recall score
3. Use GridsearchCV to tune the best baseline model for recall score whilst continuing to use the best resampling technique
4. Alter the decision threshold of the model to maximize recall score



## Step 1: Implement Multiple Resampling Methods
Thank goodness for the Imbalanced-learn library because it makes data resampling much easier. The term *resampling* refers to the practice of balancing data by either selecting a subset of the majority class equal in size to the minority class (known as under-sampling) or making the minority class as large as the majority (known as over-sampling). There are many different techniques for doing these processes, but we'll try out the following:

* **Random Over-sampling** - Randomly selects samples from the minority class and replaces them in the dataset.
* **Random Under-sampling** - Randomly selects and removes samples from the majority class.
* **SMOTE Over-sampling** - Synthetic Minority Over-sampling Technique generates new samples for the minority class by imitating its features. The original paper explaining the technique can be found [here](https://arxiv.org/abs/1106.1813).
* **Cluster Under-sampling** - Under samples the majority class by replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm.


```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import fbeta_score, recall_score, accuracy_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
import numpy as np

def rand_oversample(X, y):
  ros = RandomOverSampler(random_state=42)
  X_res, y_res = ros.fit_resample(X, y)
  return {'X_train': X_res, 'y_train':y_res}

def rand_undersample(X, y):
  rus = RandomUnderSampler(random_state=42)
  X_res, y_res = rus.fit_resample(X, y)
  return {'X_train': X_res, 'y_train':y_res}

def smote_oversample(X,y):
  sm = SMOTE(random_state=42)
  X_res, y_res = sm.fit_resample(X, y)
  return {'X_train': X_res, 'y_train':y_res}

def cluster_undersample(X, y):
  cc = ClusterCentroids(
    estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42)
  X_res, y_res = cc.fit_resample(X, y)
  return {'X_train': X_res, 'y_train':y_res}
```

### Holdout test set
Let's split our data into a train and test set. We will not alter the balance of the test set because we won't be able to do that in the real world. However, our train set will undergo resampling as we find the best baseline model and resampling combo.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def show_imb(label):
  imb = Counter(label)
  return str(imb[0]) + ' Negative Samples | ' + str(imb[1]) + ' Positive Samples'

vectorizer = TfidfVectorizer()
vect_df = pd.DataFrame(vectorizer.fit_transform(df['clean_review']).todense(), columns=vectorizer.get_feature_names_out())
vect_df['label'] = df['label'].copy()

df_train, df_test = train_test_split(vect_df, test_size = .1, random_state = 0)

y_train, y_test = df_train['label'].tolist(), df_test['label'].tolist()
X_train, X_test = df_train.drop(['label'], axis = 1), df_test.drop(['label'], axis = 1)

print('Test Set Class Balance: ' + show_imb(y_test))
```

    Test Set Class Balance: 872 Negative Samples | 52 Positive Samples


## Step 2: Train Multiple Baseline Classifiers Using Resampling Methods
Let's alter our training data using each of the resampling methods and train with it. We'll also train using the unsampled original training data to act as a baseline.


```python
sampled_data_dict = {'no_sample':           {'X_train': X_train, 'y_train':y_train},
                     'random_oversample':   rand_oversample(X_train, y_train),
                     'random_undersample':  rand_undersample(X_train, y_train),
                     'smote_oversample':    smote_oversample(X_train, y_train),
                     'cluster_undersample': cluster_undersample(X_train, y_train)}

# A look at the class balance following resampling
for s in sampled_data_dict.keys():
  txt = 'Class Balance with {:<25}' + show_imb(sampled_data_dict[s]['y_train'])
  print(txt.format(s + ':'))
```

    Class Balance with no_sample:               7767 Negative Samples | 547 Positive Samples
    Class Balance with random_oversample:       7767 Negative Samples | 7767 Positive Samples
    Class Balance with random_undersample:      547 Negative Samples | 547 Positive Samples
    Class Balance with smote_oversample:        7767 Negative Samples | 7767 Positive Samples
    Class Balance with cluster_undersample:     547 Negative Samples | 547 Positive Samples



```python
from tqdm.notebook import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Baseline Models
models_dict = {'Naive Bayes':         GaussianNB(),
               'Random Forest':       RandomForestClassifier(max_depth=10, random_state=0),
               'Logistic Regression': LogisticRegression(random_state=0)}

# Dataframe to view test results
results_df = pd.DataFrame(columns = ['model_name', 'sampling_type', 'model', 'fbeta', 'recall'])

# Iterate over all models and all sampling techniques
for m in tqdm(models_dict.keys()):
  for d in tqdm(sampled_data_dict.keys()):
    # Fit model on resampled data
    model = models_dict[m].fit(sampled_data_dict[d]['X_train'], sampled_data_dict[d]['y_train'])

    # Predict on unaltered test set
    preds = model.predict(X_test)

    # Add test results to dataframe
    results = [m, d, model, fbeta_score(y_test, preds, beta = 2), recall_score(y_test, preds)]
    results_df.loc[-1] = results
    results_df.index = results_df.index + 1
    results_df = results_df.sort_index()

results_df.sort_values(by = 'recall', ascending = False)
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model_name</th>
      <th>sampling_type</th>
      <th>model</th>
      <th>fbeta</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Random Forest</td>
      <td>cluster_undersample</td>
      <td>(DecisionTreeClassifier(max_depth=10, max_feat...</td>
      <td>0.442708</td>
      <td>0.980769</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>cluster_undersample</td>
      <td>LogisticRegression(random_state=0)</td>
      <td>0.634518</td>
      <td>0.961538</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression</td>
      <td>random_undersample</td>
      <td>LogisticRegression(random_state=0)</td>
      <td>0.664820</td>
      <td>0.923077</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Random Forest</td>
      <td>random_undersample</td>
      <td>(DecisionTreeClassifier(max_depth=10, max_feat...</td>
      <td>0.574413</td>
      <td>0.846154</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Logistic Regression</td>
      <td>random_oversample</td>
      <td>LogisticRegression(random_state=0)</td>
      <td>0.684713</td>
      <td>0.826923</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Random Forest</td>
      <td>random_oversample</td>
      <td>(DecisionTreeClassifier(max_depth=10, max_feat...</td>
      <td>0.598291</td>
      <td>0.807692</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression</td>
      <td>smote_oversample</td>
      <td>LogisticRegression(random_state=0)</td>
      <td>0.665584</td>
      <td>0.788462</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Naive Bayes</td>
      <td>cluster_undersample</td>
      <td>GaussianNB()</td>
      <td>0.288462</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Naive Bayes</td>
      <td>smote_oversample</td>
      <td>GaussianNB()</td>
      <td>0.286344</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Naive Bayes</td>
      <td>random_oversample</td>
      <td>GaussianNB()</td>
      <td>0.283843</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Naive Bayes</td>
      <td>no_sample</td>
      <td>GaussianNB()</td>
      <td>0.283843</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Random Forest</td>
      <td>smote_oversample</td>
      <td>(DecisionTreeClassifier(max_depth=10, max_feat...</td>
      <td>0.523649</td>
      <td>0.596154</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Naive Bayes</td>
      <td>random_undersample</td>
      <td>GaussianNB()</td>
      <td>0.341463</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Logistic Regression</td>
      <td>no_sample</td>
      <td>LogisticRegression(random_state=0)</td>
      <td>0.115207</td>
      <td>0.096154</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Random Forest</td>
      <td>no_sample</td>
      <td>(DecisionTreeClassifier(max_depth=10, max_feat...</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>

In terms of recall the random forest / cluster undersampling combo is the best, but it is quite weak in terms of F2 score, which means it will likely give us a lot of false positives in its predictions. The logistic regression / cluster undersample performs similarly in terms of recall, but maintains a much better F2 score, so we'll move forward with that model! Let's have a better look at its performance out of the box using a confusion matrix.


```python
def show_confusion(y_test, pred):
  cf = confusion_matrix(y_test, pred)
  group_names = ["True Negatives","False Positives","False Negatives","True Positives"]
  group_perc = [cf.flatten()[0] / (cf.flatten()[0] + cf.flatten()[1]), cf.flatten()[1] / (cf.flatten()[0] + cf.flatten()[1])
                  , cf.flatten()[2] / (cf.flatten()[2] + cf.flatten()[3]), cf.flatten()[3] / (cf.flatten()[2] + cf.flatten()[3])]
  group_perc_str = ["{:.0%}".format(each) for each in group_perc]
  group_counts = ["{0:0.0f}".format(value) for value in cf.flatten()]
  labels = labels = [f"{v1} {v2}\n{v3} of class total" for v1, v2, v3 in zip(group_counts, group_names, group_perc_str)]
  labels = np.asarray(labels).reshape(2,2)
  group_perc = np.asarray(group_perc).reshape(2,2)
  print(sns.heatmap(group_perc, annot=labels, fmt = '', cmap='Greens'))

best_baseline_model = results_df.loc[(results_df['model_name'] == 'Logistic Regression') & (results_df['sampling_type'] == 'cluster_undersample')]['model'][0]
show_confusion(y_test, best_baseline_model.predict(X_test))
``` 
![png](/base_cf.png)
    


We're already in a really good spot as we are correctly identifying the positive class over 96% of the time. We have more false positives than I'd like, but maybe some hyperparameter tuning can help here.

## Step 3: Finding Best Hyperparams with Grid Search
For finding the best logistic regression hyperparams, we will use Scikit-Learn's GridSearchCV in combination with an Imblearn Pipeline. GridsearchCV performs an exhaustive search over the provided parameters and performs cross-validation for every parameter combination. Imblearn's Pipeline allows us to perform the same resampling technique we chose in step two with grid search. The pipeline will resample the training data using cluster undersampling but will not touch the balance of the test data of the fold. We'll setup GridsearchCV to optimize for recall score and have it return the model that performed the best in terms of average recall across all folds.  


```python
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

model = Pipeline([
        # Perform resampling on train split
        ('sampling', ClusterCentroids(estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42)),
        ('classification', LogisticRegression(random_state = 42))
    ])

parameters = {'classification__solver':('liblinear', 'lbfgs'), 'classification__C':[1, 5, 10]}

# Returns best model in terms of recall score
best_grid_model = GridSearchCV(model, param_grid = parameters, cv = 5, refit = True, scoring = 'recall')
best_grid_model.fit(X_train, y_train)

# A look at how well all hyperparameter combinations perform
pd.DataFrame(best_grid_model.cv_results_).sort_values(by = 'rank_test_score')
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_classification__C</th>
      <th>param_classification__solver</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>29.715823</td>
      <td>3.306074</td>
      <td>0.068921</td>
      <td>0.016334</td>
      <td>5</td>
      <td>liblinear</td>
      <td>{'classification__C': 5, 'classification__solv...</td>
      <td>0.908257</td>
      <td>0.935780</td>
      <td>0.936364</td>
      <td>0.909091</td>
      <td>0.889908</td>
      <td>0.915880</td>
      <td>0.017857</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29.644415</td>
      <td>3.598677</td>
      <td>0.095378</td>
      <td>0.026857</td>
      <td>5</td>
      <td>lbfgs</td>
      <td>{'classification__C': 5, 'classification__solv...</td>
      <td>0.908257</td>
      <td>0.935780</td>
      <td>0.936364</td>
      <td>0.909091</td>
      <td>0.889908</td>
      <td>0.915880</td>
      <td>0.017857</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>29.187842</td>
      <td>2.767306</td>
      <td>0.057547</td>
      <td>0.019141</td>
      <td>1</td>
      <td>liblinear</td>
      <td>{'classification__C': 1, 'classification__solv...</td>
      <td>0.899083</td>
      <td>0.935780</td>
      <td>0.927273</td>
      <td>0.900000</td>
      <td>0.889908</td>
      <td>0.910409</td>
      <td>0.017804</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.689645</td>
      <td>3.572646</td>
      <td>0.104958</td>
      <td>0.027626</td>
      <td>1</td>
      <td>lbfgs</td>
      <td>{'classification__C': 1, 'classification__solv...</td>
      <td>0.899083</td>
      <td>0.935780</td>
      <td>0.927273</td>
      <td>0.900000</td>
      <td>0.889908</td>
      <td>0.910409</td>
      <td>0.017804</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28.605801</td>
      <td>2.047266</td>
      <td>0.052858</td>
      <td>0.005150</td>
      <td>10</td>
      <td>liblinear</td>
      <td>{'classification__C': 10, 'classification__sol...</td>
      <td>0.908257</td>
      <td>0.926606</td>
      <td>0.918182</td>
      <td>0.890909</td>
      <td>0.889908</td>
      <td>0.906772</td>
      <td>0.014572</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>32.235926</td>
      <td>3.819624</td>
      <td>0.099514</td>
      <td>0.020922</td>
      <td>10</td>
      <td>lbfgs</td>
      <td>{'classification__C': 10, 'classification__sol...</td>
      <td>0.908257</td>
      <td>0.926606</td>
      <td>0.918182</td>
      <td>0.890909</td>
      <td>0.889908</td>
      <td>0.906772</td>
      <td>0.014572</td>
      <td>5</td>
    </tr>
  </tbody>
</table>


A look at the confusion matrix following hyperparameter tuning.


```python
show_confusion(y_test, best_grid_model.predict(X_test))
```
    
![png](/hyperparam_cf.png)
    


So, this is bit better. Our false positive rate has reduced slightly, meaning that this new model is having an easier time differentiating between the two classes than before. In many cases this would suffice, but what if we needed to attempt to let no positive samples to slip past our classifier? This is where the fourth and final step could come in, altering the decision threshold of the model.

## Step 4: Altering the Decision Threshold
Unfortunately, boosting the recall of our model will come at a cost.


```python
from sklearn.metrics import precision_recall_curve
y_scores = best_grid_model.predict_proba(X_test)[:, 1]
p, r, thresholds = precision_recall_curve(y_test, y_scores)
viz_df = pd.DataFrame({'Precision':p, 'Recall':r})
sns.lineplot(data = viz_df, x = "Recall", y = "Precision")
```
  
![png](/pr_curve.png)
    



```python
def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]

init_threshold = 0.5
pred = adjusted_classes(y_scores, init_threshold)
recall = recall_score(y_test, pred)

while recall != 1.0:
  init_threshold = init_threshold - 0.001
  pred = adjusted_classes(y_scores, init_threshold)
  recall = recall_score(y_test, pred)

print("Maximum Threshold for Perfect Recall:", init_threshold)
show_confusion(y_test, pred)
```

    Maximum Threshold for Perfect Recall:  0.3679999999999999
    
![png](/perfect_recall_cf.png)
