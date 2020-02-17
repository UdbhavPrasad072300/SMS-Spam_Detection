#!/usr/bin/env python
# coding: utf-8

# # SMS Spam Detection using NLP <br><br>
Written by: Udbhav Prasad <br>
Linkedin: https://www.linkedin.com/in/udbhav-prasad-1506b7192/ <br>
HackerRank: https://www.hackerrank.com/uprasad1 <br>
Github: https://github.com/UdbhavPrasad072300 <br>
Computer Science Co-op - Ryerson University <br> <br> <br> <br>Using nltk and sklearn to build a classifier to determine whether text is an SMS or SPAM(labeled "ham" for SMS and "spam" for SPAM)We use the SMS Spam Collection Dataset from Kaggle: (https://www.kaggle.com/uciml/sms-spam-collection-dataset) <br>
<br>
To get a sense of the data we first see the first ten lines: <br>
# ## Step 1: Reading Data

# In[1]:


import csv
from itertools import islice 

with open('spam.csv', 'r') as f: #reading csv file
    reader = csv.reader(f)
    spam = list(reader)

spam = spam[1:]
    
for row in islice(spam, 10):
    print(row)


# ## Step 2: Cleaning up Data
There are empty values at the end of each rows which needs to be handled (last three columns): 
# In[2]:


truncatedSpam = [row[0:2] for row in spam]
for row in islice(truncatedSpam, 10):
    print(row)

We separate words with punctuation then filter the words to remove punctuation:
# In[3]:


import re
import string

def remove_punctuation_and_lowercase(string1):
    string1 = "".join([char for char in string1 if char not in string.punctuation])
    return(string1.lower())

for i in range(1, len(truncatedSpam)):
    truncatedSpam[i][1] = truncatedSpam[i][1].split()
    
for row in range(1, len(truncatedSpam)):
    for word in range(0, len(truncatedSpam[row][1])):
        truncatedSpam[row][1][word] = remove_punctuation_and_lowercase(truncatedSpam[row][1][word])

for i in islice(truncatedSpam, 10):
    print(i)

now we need to remove stopwords: <br>
These are the stopwords in english which are in the nltk package:
# In[4]:


import nltk

stopwords = nltk.corpus.stopwords.words("english") #stopwords in english from nltk package
print(stopwords)

we now remove the stopwords from our data:
# In[5]:


def remove_stopwords(textList):
    temp = [word for word in textList if word not in stopwords]
    return temp

for row in range(1, len(truncatedSpam)):
    truncatedSpam[row][1] = remove_stopwords(truncatedSpam[row][1])

for i in islice(truncatedSpam, 10):
    print(i)

Lemmatizing is important to get generalize words to their base meaning(singular, plural, past tense, present tense, future tense) <br>
<br>
Lemmatizing is slower than stemming but it much better for your classifier (we got all the time in the world right now,  as im probably going to upload this as a HTML file) <br>
<br>
We now lemmatize our words:
# In[6]:


wn = nltk.WordNetLemmatizer()

for row in range(1, len(truncatedSpam)):
    for word in range(0, len(truncatedSpam[row][1])):
        truncatedSpam[row][1][word] = wn.lemmatize(truncatedSpam[row][1][word])

for i in islice(truncatedSpam, 10):
    print(i)


# ## Step 3: Vectorizing Data
Creating a sparse matrix for classifier (machine learning):
# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer

#truncatedSpam = truncatedSpam[1:]

tfidf_vect = TfidfVectorizer([i[1] for i in truncatedSpam])
X_counts = tfidf_vect.fit_transform([i[1] for i in spam])
print(X_counts)


# tfidf vectorizer makes a vector based on the frequency of each possible word with a weighted value(weighting is based on values that seem to be a determining factor)
# 
# There are other alternatives like countVectorizing

# ## Step 4: Building the Classifier

# In[8]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd

X_features = pd.DataFrame(X_counts.toarray()) #create dataframe for random forest
#print(X_features.head(1))

rf = RandomForestClassifier(n_jobs=-1)
k_fold1 = KFold(n_splits=5)

k_fold2 = KFold(n_splits=10)

print(cross_val_score(rf, X_features, [i[0] for i in truncatedSpam], cv=k_fold1, scoring='accuracy', n_jobs=-1))
print('LINE BREAK'.center(100,'-'))
print(cross_val_score(rf, X_features, [i[0] for i in truncatedSpam], cv=k_fold2, scoring='accuracy', n_jobs=-1))

Using k-fold cross validation for random forest classifier with k = 5 and k = 10, <br>
Basically means that it splits the dataset into 10 and 5 pieces and evaluates on each piece with a classifier built from the remaining pieces
# ## Holdout Sets

# In[9]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_features, [i[0] for i in truncatedSpam], test_size=0.2)

temp = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)

temp.fit(X_features, [i[0] for i in truncatedSpam]) #fit by data and their corresponding tags

print(temp.predict_proba(X_features.head())) #head gives the first 5 rows, so prediction happens for the first five rows

print(X_features.head())

print([truncatedSpam[i][0] for i in range(0, 5)])

#temp_model = temp.fit(X_train, Y_train)
#sorted(zip(temp_model.feature_importances_, X_train.columns), reverse=True)[0:10]


# # Conclusion

# ## The two columns above represent the probability of being either ham/spam
# In this case, the first one has probability of being ham by 95.54 percent (as well as the second, fourth and fifth).
# The third value has 75 percent probability of being spam and 25 percent being ham.
# Below the result i have shown the data and the tags for it (to verify that it is correct).
In conclusion, it has many flaws and shortcomings such as (there are most likely more): <br>
# Feature Engineering (adding length of messages, amount of punctuation) 

# Testing can be very much improvement with external data

# Many better methods of doing this, using different values or methods all together

# Better/More data cleaning methods could be used 

# Holdout sets are never an indicator of quality (well, maybe a little, but negligible)
