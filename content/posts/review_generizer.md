---
title: Using Unsupervised ML to 'Generize' Product Reviews
date: 2023-09-30
author: Chandler Underwood
description: Using NLP and unsupervised machine learning techniques, I create a model that can select the most typical reviews containing a keyword.
ShowToc: true
TocOpen: true
---

# Motivation
In my last post where I scraped reviews for Crocs Clogs, I mentioned that I often find myself wishing for a succinct summary of the reviews for a product. Let's flesh that out a bit more. What I mean when I say "succinct summary" is that I want a quick understanding of a specific aspect of a given product. For example, I know that crocs come in amazing colors already. I can see that in the photos. But, how do they fit? What about their durability? I find myself often most concerned with a specific aspect of a product such as those. I want to know what people are typically saying about fit and durability. Many retailers offer a search bar for reviews, so you can filter reviews on a keyword. BUT, searching for "fit" across all crocs reviews would return a ton of samples, and how can we know which ones are representative of the general sentiment people have in regards to fit? What if we could give consumers a snapshot of the reviews containing a word or phrase they search for? Could we show them a small set of reviews that best represent all the reviews that mention the word "fit", for example? I think we can!

# How to Make it Happen
Did someone say clustering??? Because they would be correct. Unsupervised clustering of the reviews will allow us to find the most typical subset, and *k*-means will be very helpful here. 

According to Wikipedia - *k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster.*

In other words, *k*-means iteratively searches for the best representation of each cluster (the center) and assigns samples to a cluster based on their distance from each cluster center. So, we can say that a cluster's center is an approximation of all the cluster's members. Following this line of thinking, if we fit a *k*-means model to some data and only ask it to find 1 cluster, the cluster center will act as a prototype for all the data that was passed to the model. To find the most representative subset of reviews for a particular keyword, we can filter the reviews based on the keyword, find a cluster center for the samples, and get the X closest samples to center. The closest samples to the center will be the most representative of the population. If this isn't clear, check out the GIF below and imagine we're only trying to make one cluster. Think about how the centroid would move in that scenario.

![Scenario 1: Across columns](/keans.gif)
Credit - https://towardsdatascience.com/clear-and-visual-explanation-of-the-k-means-algorithm-applied-to-image-compression-b7fdc547e410

# Tools Needed
Of course we will need Pandas to make for easy data manipulation and usage. We also will need to clean up our text data, so we'll use NLTK. Finally, we need a way to vectorize the text and fit a *k*-means model to it, so we'll use Scikit-Learn for that part. 

```python
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


import re
import string
```

# Clean Review Data
It's usually very important (except when fine-tuning an LM) that we stem, remove punctuation, convert to lowercase, and remove stopwords from the text we're fitting a model on. We don't want "Oranges" and "orange" to be treated as different words, nor do we need words like "the" and "to", for example. 

```python
def clean_text(text):
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

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

df = pd.read_csv('data\croc_reviews.csv')
df['clean_review'] = [clean_text(text_to_clean) for text_to_clean in df['review']]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>date</th>
      <th>rating</th>
      <th>clean_review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>!!!!!! E X C E L L E N T!!!!!!!!!!</td>
      <td>April 7, 2022</td>
      <td>5.0</td>
      <td>e x c e l l e n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"They're crocs; people know what crocs are."</td>
      <td>April 3, 2021</td>
      <td>5.0</td>
      <td>theyr croc peopl know croc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>- Quick delivery and the product arrived when ...</td>
      <td>March 19, 2023</td>
      <td>5.0</td>
      <td>quick deliveri product arriv compani said woul...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>...amazing "new" color!! who knew?? love - lov...</td>
      <td>July 17, 2022</td>
      <td>5.0</td>
      <td>amaz new color knew love love love</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0 complaints from me; this is the 8th pair of ...</td>
      <td>June 4, 2021</td>
      <td>5.0</td>
      <td>0 complaint 8th pair croc ive bought like two ...</td>
    </tr>
    <tr>
      <th>...</th>
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
      <td>definit buy mani color reason 45 materi feel t...</td>
    </tr>
    <tr>
      <th>9234</th>
      <td>I wish I would have bought crocs a long time ago.</td>
      <td>April 8, 2021</td>
      <td>5.0</td>
      <td>wish would bought croc long time ago</td>
    </tr>
    <tr>
      <th>9235</th>
      <td>wonderful. Gorgeous blue; prettier in person!</td>
      <td>April 27, 2022</td>
      <td>5.0</td>
      <td>wonder gorgeou blue prettier person</td>
    </tr>
    <tr>
      <th>9236</th>
      <td>Wonerful. Very comfy, and there are no blister...</td>
      <td>April 8, 2021</td>
      <td>5.0</td>
      <td>woner comfi blister feet unlik brand one</td>
    </tr>
    <tr>
      <th>9237</th>
      <td>Work from home - high arch need good support a...</td>
      <td>May 22, 2023</td>
      <td>5.0</td>
      <td>work home high arch need good support comfort ...</td>
    </tr>
  </tbody>
</table>
<p>9238 rows × 4 columns</p>
</div>


# Finding Typical Reviews
The workflow for finding the most typical reviews will be:
1. Filter reviews on a keyword and rating (either positive or negative)
2. Vectorize the reviews with TF-IDF
3. Fit a single *k*-means cluster to the vectorized data
4. Calculate the distance of each sample to the center of the cluster and sort

```python
def get_tfidf(text):
    vectorizer = TfidfVectorizer()

    # Fit and transform review data
    X = vectorizer.fit_transform(text)
    
    return X

def kmeans_distance(df):
    # Get a TF-IDF matrix representation of the reviews
    X = get_tfidf(df['clean_review'])

    # Fit one cluster on the data.
    kmeans = KMeans(n_clusters=1, random_state=0, n_init=100).fit(X)

    # Compute the distance for each sample to the center of the cluster
    distances = kmeans.transform(X)**2
    
    return distances
    
def find_most_typical(df, word, rating, asc = True):
    # Let's consider anything 4 stars and up a positive review
    if rating == 'positive':
        df = df[df['rating'] >= 4]
    else:
        df = df[df['rating'] < 4]

    # Clean word so it matches root word in cleaned reviews
    filter_word = ' ' + clean_text(word)  + ' '

    # Filter using clean word
    df = df[df['clean_review'].str.contains(filter_word, regex = True)]

    # Retrieve review distance to center of kmeans cluster
    df['distance_to_center'] = kmeans_distance(df)

    if asc == False:
        print("Most atypical " + rating + " reviews metioning: ", word)
    else:
        print("Most typical " + rating + " reviews metioning: ", word)
    
    # Print in descending order for distance unless atypical test flag is set to False
    for each in df.sort_values(by = 'distance_to_center', ascending = asc)['review'].tolist()[:3]:
        print('# ', each)
```

# Psuedo-evaluation
We don't really have a way to evaluate this model, so we're going to have to use some intuition! Firstly, let's look at what the 3 most typical positive reviews that mention "crocs" would say.

```python
find_most_typical(df, 'crocs', 'positive')
```

    Most typical positive reviews metioning:  crocs
    #  I love to wear these crocs because they are so comfortable.
    #  I love wearing my crocs. They are so comfortable.
    #  I love my pair of Crocs. They are so comfortable!
    
To make sure we're on the right track, what do the most *atypical* positive reviews for crocs say? We can find the samples that are farthest from the center by sorting their distance greatest to smallest.

```python
find_most_typical(df, 'crocs', 'positive', asc = False)
```

    Most atypical positive reviews metioning:  crocs
    #  Lids adore their Crocs! Fun with swapping charms and accessoryizing.
    #  Purr Nickis Impact Crocs. Ima need you to give my girl a ha check!!!!
    #  You can turn any none-croc person into a croc lover.
    
Those are definitely a bit weird! But, it looks like we've built something that works. Very cool.

# Looking at More Keywords and Ratings
Let's see what people have to say about the durability and fit of crocs that is positive.

```python
find_most_typical(df, 'durability', 'positive')
```

    Most typical positive reviews metioning:  durability
    #  I love the classic clogs. I have 4 other pairs that I've purchased over the years, and they are still in great condition. I purchased this clog for the summer. It's perfect as an everyday shoe. I use mine to the beach, park, and even kayaking. They are very easy to clean. They are comfy, durable, and super cute to wear. I love my crocs.
    #  I absolutely love the comfort and durability of Crocs. The classics are my favorite. I would highly recommend Crocs to anyone looking for a comfortable shoe.
    #  I recently purchased the Crocs Classic Clog, and I have to say, I'm impressed. These shoes are incredibly comfortable and versatile. I love that I can wear them around the house, to run errands, or even to work. The foam material of the clogs is soft and cushy, which makes them perfect for long walks or standing for extended periods. They also offer excellent arch support, which is a bonus for someone like me who struggles with plantar fasciitis. Another thing I appreciate about these clogs is how easy they are to clean. A quick rinse under the faucet or a gentle scrub with a soft brush is all it takes to get them looking like new again. Overall, I highly recommend the Crocs Classic Clog to anyone looking for a comfortable and durable pair of shoes. They may not be the most stylish footwear option out there, but they more than make up for it with their practicality and comfort.
    


```python
find_most_typical(df, 'fit', 'positive')
```

    Most typical positive reviews metioning:  fit
    #  Love the color and fit just as comfortably as my other crocs.
    #  I love these. They are a perfect fit and very comfortable. I love the color as well.
    #  Very comfortable and great fit. I love them.
    
What do they say about those aspects that are negative?

```python
find_most_typical(df, 'durability', 'negative')
```

    Most typical negative reviews metioning:  durability
    #  The best Crocs for me are the off-road styles with an adjustable heel strap; however, you no longer make these in my size (14M). These are the best fishing and boating shoes ever, comfortable, durable, cool, quick drying, and very comfortable. Since you do not make the off-road in my size, I decided to try regular crocs. They are good but would be perfect if the heel strap was adjustable and would receive a five-star rating from me. Please bring back the off-road crocs in larger sizes!
    #  Not comfortable or durable. They are overpriced for what they are. Shameful
    #  It's comfy, functional, and durable, but they really should advise you buy a size down. Mine's much too roomy.
    


```python
find_most_typical(df, 'fit', 'negative')
```

    Most typical negative reviews metioning:  fit
    #  I have over 20 pairs of crocs, and lately, the last 5 pairs I've purchased have all fit differently. I'm usually a men's 8 & recently bought a red pair of the classic clog. They were entirely too big, which made no sense because I have pink ones the same size that fit perfectly. I purchased a purple pair & decided to size down & get a men's 7 & they were way too small (which also made no sense because I have blue crocs the same size that were a more snug fit). My suggestion is to either make half sizes, or stop with this whole "Roomy Fit" thing that you all are doing. There is zero reason why each pair of crocs should have a different fit. I'll never order crocs online again. I highly recommend just going to the store to make your purchase. The return process is also very strenuous because Crocs does not offer exchanges. So now I have to send them back to the store via UPS, wait for my return to be processed, then wait until I can make my way to a crocs store because the closest store in my area is 36.6 miles away. Ridiculous.
    #  Well, the crocs do not fit my granddaughter. One is actually a different size than the other. She received another pair of Crocs from her dad, and even though the pair he got her are size 9, and the pair I got her are size 9, the pair he bought her fits and the ones I got her do not. One pair was made in China and one pair was made in Vietnam.
    #  I have had several pairs of crocs in the past, different styles and colors. Direct from the company. I have been disappointed in the consistency of the sizing. I had a size 10 in the classic style and wanted another pair in a different color. When they arrived, the fit was at least a size larger and wider than my original pair. Then I exchanged for a size 9. This fit better, but now the left shoe is smaller than the right. Not happy.
    
Well that was fun! I really like this idea and will potentially deploy a model that does this in the future...
