# Week 2


**Table of Contents**
- [Day 1](#day-1)
- [Day 2](#day-2)
- [Day 3](#day-3)

<br>

The goals of this week are to:

- understand the overall workflow of a machine learning project
- to use scikit learn to implement a supervised classifier for your project
- evaluate your approach on your labeled dataset

<br>

## Day 1

Today we will extract some features from our data and perform an initial classification experiment.

See the starter notebook: https://github.com/tapilab/elevate-osna-starter/tree/master/notebooks/W2L1.ipynb


<br>

## Day 2

Continue working on your notebook from the last lab. Do the following:

- Use [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to create a matrix of all terms. Experiment with the following to see the affect on accuracy:
  - `min_df`: [1,2,5,10]
  - `max_df`: [1, .95, .8]
  - `ngram_range`: [(1,1), (1,2), (1,3)]
- Experiment with different regularization for LogisticRegression
  - `C`: [.1, 1, 5, 10]
  - `penalty`: [l1, l2]
- Summarize your results with a table for each setting, like this:

| C  | Accuracy|
|----|---------|
| .1 | xxx     |
| 1  | xxx     |

- Vary one parameter at a time, while using the defaults for the rest.Let the defaults be (min_df=2, max_df=1, ngram_range=(1,1), C=1, penalty=l2). 

<br>

## Day 3

Today we will ...



