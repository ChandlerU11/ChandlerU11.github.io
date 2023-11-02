+++
title = "Creating a Crocs Review Summarizer"
date = "2023-08-19"
author = "Chandler Underwood"
description = "Using several NLP techniques along with the transformers library I fine-tune a T5 model to summarize reviews for Crocs Classic Clog."
+++

In case you haven't already checked-out my last post where I scraped reviews for Crocs off of Zappos, let me explain why we're here. Online shopping is pretty awesome, but I think it can be better! As someone who relies on the anecdotal information you get about a product from its reviews, I have found myself often wishing there was a short summary of all the reviews to speed up my decision process. Currently, it is quite common for companies to extract keywords from across the reviews to show consumers, but I would very much prefer to see something at a sentence-level to summarize what people generally are saying at about the product in question. 

## Finding "Summaries" for the Reviews

There is a small problem with this data we have scraped. We want to create a review summarizer, but there are no example summaries in this dataset. Typically, one would use the titles of the reviews as a proxy for summaries, but Zappos does not ask reviewers to give their reviews a title. So what can we do?

We could separate the data into reviews that are really short and reviews that are at least a sentence long. Then, using embeddings and cosine distance, find the most similar short review for each long review and use the short one as a summary. I tried this at first, and the results were less than stellar...We seriously lacked diversity in the summaries, so the output of the summarizer collapsed immediately. 

After some more trial and error trying to make this method work, I thought of a different approach. We could make a "generic" conditional review generator instead. This can be acheived by passing a low tempurature value to the generator, forcing the generator to only output sequences it is very confident in. If the output of this generator is truly generic, it can be thought of as a pseudo-summarizer for the reviews, given that we pass inputs to it that are representative of the dataset. These high-level inputs will be a review's rating and some the most common keywords. 

## Data Discovery and Processing
As I'm sure you can imagine, a lot of the reviews contain some grammar issues, and we don't want our generator to spit out any gibberish. To fix this we can use the *grammar_corrector* from grammarly. The *grammar corrector* will automatically capitilize and add punctuation to text for us, so we can strip the puncuation and covert the reviews to lowercase prior to grammar correction. Preprocessing the text like this will allow us to prune reviews that are extremely short and give us a more uniform output from the *grammar_corrector*. 

```python
from nltk.tokenize import RegexpTokenizer
import re

df = pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/croc_reviews_total.csv')

tokenizer = RegexpTokenizer(r'\w+')

# Remove all punctuation and lowercase text
df['review_clean'] = [re.sub(r'[^\w\s]', '', text).lower() for text in df['review']]

# New feature to describe how many words are in review
df['review_len'] = [len(tokenizer.tokenize(s)) for s in df['review_clean']]

# Only keep reviews containing at least 5 words
df = df[df['review_len'] >= 5]

```

And the code to correct the grammar of processed review text...

```
tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
model = AutoModelForSeq2SeqLM.from_pretrained("grammarly/coedit-large").to(device)

def tokenize_function(examples):
    return tokenizer(examples['rev'], padding='max_length', truncation=True)

train_split = np.array_split(df['review_clean'].replace('\n', ''), 500)

review_para = []
for each in tqdm(train_split):
  inputs = tokenizer(list(each), return_tensors="pt", padding = True, truncation = True)
  response = model.generate(inputs["input_ids"].to(device), max_length =  500)
  response = [tokenizer.decode(each, skip_special_tokens = True) for each in response]
  review_para.extend(response)

df['review_para'] = review_para
df[['review', 'review_para']].head()
```
Sweet (I think I'm in love with Huggingface <3).

## Input Construction

Now, we need to find our keywords to use as input for training. We'll reuse this function in the end to find our most commonly keywords. SpaCy's POS (part-of-speech) tagging is a great option to use here as it will allow us to pull out words based on their tag. I believe adjectives and verbs are the most important types of words used to describe footwear. Those two parts of speech are going to cover words like "small", "big", "wear", "fit", etc. 
```python
import spacy
from string import punctuation

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1500000

def get_keywords(text):
    result = []
    pos_tag = ['VERB','ADJ'] 
    doc = nlp(text.lower())
    doc = nlp(' '.join([token.lemma_ for token in doc]))
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result

keywords = []
for each in tqdm(df['para_review']):
  keywords.append(set(get_keywords(each)))

df['keywords'] = keywords

input_list = []
for i in range(len(df)):
  input_list.append(df.loc[i]['rating'].astype(str)[0] 
  + ' star review mentioning - ' + ', '.join(list(df.loc[i]['hot_words'])) + ': ')

df['input'] = input_list

df[['input', 'review_para']].head()
```
## Fine-Tuning T5


## Final Product

## Going Forward
I really like this idea and I'd like to deploy something like this.