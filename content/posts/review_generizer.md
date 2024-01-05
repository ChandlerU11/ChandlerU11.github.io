+++
title = "Using Unsupervised ML to 'Generize' Product Reviews"
date = "2023-10-10"
author = "Chandler Underwood"
description = "Using various NLP and unsupervised machine learning techniques, I create a model that can select the most typical reviews containing a set of key words."
+++

# Motivation
In my last post where I scraped reviews for Crocs Clogs, I mentioned that I often find myself wishing for a succinct summary of all the reviews for a product. Let's flesh that out a bit more. What I mean when I say "succinct" is that I want something that gives a quick and comprehensive understanding of a specific aspect of a given product. For example, I know that crocs come in amazing colors already. I can see that in the photos. But, how do they fit? What about their durability? I find myself often most concerned with a specific aspect of a product such as those. I want to know what people are typically saying about fit and durability. Many retailers offer a search bar for reviews, so you can filter reviews on a keyword. BUT, searching for "fit" across all crocs reviews could return 1k plus reviews. What if we could give consumers a snapshot of the reviews containing a word or phrase they search for? Could we show them a small set of reviews that best represent all the reviews that mention the word "fit", for example? I think we can!

