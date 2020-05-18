"""
Created on Sun May 17 17:45:14 2020

@author: Shubham Patil
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import nltk
import re
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Natural language preprocessing -------------------->

words = []
classes = []
documents = []
corpus_x = [] # to create independent variables
corpus_y = [] # to create dependent variables

json_file = open("intents.json").read()
intents = json.loads(json_file)

for intent in intents['intents']: # tokenization
    for pattern in intent['patterns']:
        pattern = re.sub('[^a-zA-Z\s]', '', pattern)
        pattern = pattern.lower()
        word = pattern.split(' ')
        ps = PorterStemmer()
        word = [ps.stem(w) for w in word if not w in set(stopwords.words('english'))]
        words.extend(word) # extended... not appended
        documents.append((word, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        
        corpus_x.append(' '.join(word))
        corpus_y.append(intent['tag'])

words = sorted(list(set(words))) 
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

from sklearn.feature_extraction.text import CountVectorizer # creates bag of words
cv_input = CountVectorizer(max_features = 1500, token_pattern=u"(?u)\\b\\w+\\b")
cv_output = CountVectorizer(max_features = 1500, token_pattern=u"(?u)\\b\\w+\\b")

x = cv_input.fit_transform(corpus_x).toarray()
y = cv_output.fit_transform(corpus_y).toarray()

x = x.tolist() # tolist() to convert internal variables to the list as well 
y = y.tolist()

# shuffle two lists with same order ( zip(): <zip two list together> + shuffle() <shuffle list> + zip(*) <unzip lists>
temp = list(zip(x, y)) # zip
random.shuffle(temp) # shuffle
x, y = zip(*temp) # unzip, creates x and y as tuples
x, y = list(x), list(y) # convet to list from tuples

# Creating Deep Learning Model ------------>

# creating layers
model = Sequential()
model.add(Dense(units = 140, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(x[0])))
model.add(Dropout(0,5))
model.add(Dense(units = 120, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0,5))
model.add(Dense(units = len(y[0]), activation = 'softmax'))

# compiling model
sgd = SGD(learning_rate = 0.05, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# fitting and saving the model
HDF = model.fit(np.array(x), np.array(y), batch_size = 5, epochs = 500, verbose = 1) # Heirarchical Data Format
model.save('CB_Model.h5', HDF)
