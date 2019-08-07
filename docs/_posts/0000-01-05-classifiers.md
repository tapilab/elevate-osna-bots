---
layout: slide
title: "Classifiers and Features"
---

We compared three classifiers:

|                           | F1       | Precision |  Recall   | 
|---------------------------|----------|-----------|-----------|
| Logistic Regression       | 0.91     |  0.91     |    0.91   |          
| Multi-layer Perceptron    | 0.84     |  0.84     |    0.84   |      
| Random Forest             | 0.89     |  0.89     |    0.89   |           

The most predictive features of bots were:
1. **Most common trigram**, bots tend to tweet the same contents.
2. **Default profile**, indicates that user has not altered the background of their profile. A high percentage of bots use default profile.
3. **Statuses count**, the number of tweets (including retweets) issued by the user. Since many bots are used to spread fake news or something, bots have a higher statuses count.

And the most predictive features of humans were:
1. verified
2. followers_count
3. tweets_avg_mentions
