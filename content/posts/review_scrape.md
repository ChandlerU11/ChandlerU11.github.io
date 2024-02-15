---
title: Building a Product Reviews Webscraper
date: 2023-11-20
author: Chandler Underwood
description: Using the Python Requests and Beautiful Soup libraries, I construct a webscraper that can farm product reviews off of Zappos.com.
ShowToc: true
TocOpen: true
---

# Motivation
Do you ever find yourself pouring over a product's reviews trying to decide if it's right for you? I sure do. Many of these times I've wished there was a quick summary of the reviews I could read to speed up the decision process. I've yet to see any online retailers doing exactly what I'm looking for, so I've decided to make my own review summarizer. But, we need some data for training such a tool.

For this project I wanted to use data that was fun, interesting, and maybe even a little contraversial... After much thought and deliberation over what product reviews to train a summary generator on, I landed on Crocs. Yes, I'm talking about the clunky rubber footwear my grandmother wears while gardening that some fashionistas would consider an abomination. What's more fun than that??? Maybe or maybe not to your surprise, I searched the internet for a dataset containing reviews for Crocs to no avail. So, we're going to have to find our own data. Follow along as I build a webscraper to farm reviews for Crocs Classic Clog, so we can train a review summarizer for the most loved and hated shoes on the planet! 

# What tools can we use? 
There are several Python libraries that are helpful for webscraping:
- **Requests:** To send HTTP requests to websites.
- **Beautiful Soup:** For parsing HTML and XML documents.
- **Selenium:** For scraping dynamic web pages using browser automation.
- **Pandas:** To manipulate and store scraped data.

In this project, I'll be using Requests, Beautiful Soup, and Pandas. Selenium isn't needed because we don't need to click anything to move through the review pages on Zappos.com as you'll see later.  

# Getting Started
Before we start scraping, we need to do some research, meaning that we need to go look around on the internet and assess potential sites for scraping. What sites would have a lot of product reviews for Crocs? The Crocs website itself, Amazon, and Zappos immediately come to mind. After following some initial site evaluation steps outlined [here](https://towardsdatascience.com/web-scraping-basics-82f8b5acd45c#:~:text=You%20can%20look%20at%20the,you%20to%20scrape%20the%20website.), I decided Zappos.com was the best place to scrape from. 

First, we need to figure out how to traverse the pages of reviews on Zappos.com. I went and clicked through the review pages and payed close attention to the URL. I noticed that it explicitly changes with each page. For example, https://www.zappos.com/product/review/7153812/page/1/orderBy/best takes you to the first page, and https://www.zappos.com/product/review/7153812/page/2/orderBy/best takes you to the second page. Easy. To traverse the pages, all we need to do is change a single character in the URL. 

# Some Code to Download Whole Webpages
I prefer to download whole webpages and extract data from them later. Why? Because comapnies oftentimes aren't super happy about you scraping their data, and you have to be carefull about how your traffic looks as you surf pages. Manners are important! Extracting data from HTML is a messy process that requires developement, and you don't want to risk having to rerun your scraper becuase you forgot to write the Beatiful Soup code to extract a product's rating, for example. A script to traverse and download Zappos review pages is below.

```python
from bs4 import BeautifulSoup
import requests
import random
import time

HEADERS = ({'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36', 'From': 'chandler_underwood@yahoo.com'})

#Download raw HTML
def getdata(url):
	r = requests.get(url, headers=HEADERS, timeout=5)
	return r.text

#Convert HTML to parsable string
def html_code(url):
	htmldata = getdata(url)
	soup = BeautifulSoup(htmldata, 'html.parser')
	return (soup)

text = "https://www.zappos.com/product/review/7153812/page/{insert}/orderBy/best"

#Iterate over all posible URLs
for i in range(1, 1000):
    #Retrieve HTML
	print("Scraping: " + text.format(insert = str(i)))
	code = html_code(text.format(insert = str(i)))

    #Save file with unique name
	file = open("path_to_folder/Zappos_Croc_File_" + str(i) + ".html","w")
	file.write(str(code))
	file.close()
  
    #Always remember to be polite
	time.sleep(random.randint(15,40))
```

# Extracting Data from Webpages
Now that we've downloaded the pages of reviews, we need to figure out how to extract the text from the HTML in a useful format. Google Chrome's page inspect tool is great here as it allows us to click on items within the webpage and find their respective HTML tagnames. 
![Scenario 1: Across columns](/Zappos_Insp.PNG)

As I clicked on the items I wanted to extract on the webpage, I took note of the tagnames associated with them. Beautiful Soup uses these tagnames to extract the text we need. Through some trial and error, I ended up with the following function to extract the desired text to make the reviews dataset. 

```python 
def cus_rev_zappos(soup, b = True):
  review_text = []
  date = []
  rating = []
  rating_type = []
  rating_return = []
  
  #Extract reivew
  for item in soup.find_all("div", class_="Xl-z Yl-z"):
      review_text.append(item.get_text())

  #Extract review date
  for item in soup.find_all("time"):
      date.append(item.get_text())

  #Extract product rating
  for item in soup.find_all("span", class_="HQ-z"):
      rating.append(item.get_text())

  #Extract product rating type (Overall, Style, Fit, etc.)
  for item in soup.find_all("em", class_="Ql-z"):
      rating_type.append(item.get_text())

  #We only care about overall product ratings for this project
  rating = list(zip(rating[5:], rating_type))
  for x in rating:
      if x[1] == 'Overall':
         rating_return.append(x[0])

  return review, date, rating_return
  ```

I then iterated over and extracted the text from the downloaded webpages using the following.  

```python 
from tqdm.notebook import tqdm

reviews, ratings, dates = [], [], []

for x in tqdm(range(len(filenames))):
  f = open("path_to_folder/" + filenames[x], 'r')
  soup = BeautifulSoup(f.read(), 'html.parser')
  review, date, rating = cus_rev_zappos(soup)
  reviews.extend(review)
  ratings.extend(rating)
  dates.extend(date)

review_df = pd.DataFrame({'review': reviews, 'rating': ratings, 'date': date})

review_df.to_csv("Zappos_Croc_Reviews_Total.csv")
```

Soon after extracting from the webpages, I found that Zappos does not publish reviews after the 400th page :(. We got a decent amount of data, however! We ended up with over 9,000 distinct reviews for Crocs.