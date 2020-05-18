"""
Created on Mon May 18 10:44:25 2020

@author: Shubham Patil
"""

from tkinter import *
import nltk
import re
import pickle
import json
import random
import numpy as np
from keras.models import load_model
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

model = load_model('CB_Model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


class GUI(object):
    
    def __init__(self):
        
        self.window = Tk()
        self.window.geometry("900x650")
        self.window.resizable(0, 0)
        self.window.config(bg = '#000')
        
        self.top_frame = Frame(self.window, bg = '#000')
        self.bottom_frame = Frame(self.window, bg = '#273443')
        self.top_frame.pack(side = TOP)
        self.bottom_frame.pack(side = TOP)
        
        self.scroll = Scrollbar(self.top_frame, bd = 0)
        self.scroll.pack(side = RIGHT, fill = Y, pady = 2)
        
        
        self.chatBox = Text(self.top_frame, yscrollcommand = self.scroll.set, width = 100, height = 30, bd = 0, font = ('Arial', 10))
        self.chatBox.pack(side = LEFT, pady = 2)
        
        self.scroll.config(command = self.chatBox.yview)
        
        self.entry = Text(self.bottom_frame, width = 55, bd = 0, font = ('Arial', 15))
        self.entry.pack(side = LEFT)
        
        self.send_button = Button(self.bottom_frame, text = "send", bd = 0, bg = '#34b7f4', fg = '#fff', width = 10, height = 20, font = ('Arial', 15), command = self.send_text)
        self.send_button.pack(side = LEFT)
        
        self.window.mainloop()
        
    def tokanize(self, text):
        text = re.sub('[^a-zA-Z\s]', '', text)
        text = text.lower()
        text = text.split(' ')
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text]
        text = ' '.join(text)
        return text
    
    def bag_of_words(self, text):
        cv = CountVectorizer(max_features = 15000, token_pattern=u"(?u)\\b\\w+\\b")
        cv.fit_transform(words).toarray()
        input_array = cv.transform([text]).toarray()
        return input_array
        
    def get_response(self, text):
        tokenised_text = self.tokanize(text)
        input_array = self.bag_of_words(tokenised_text)
        probability = model.predict(input_array).tolist()[0] # list of probabilities
        index = probability.index(max(probability)) # tag with high probability
        
        for ints in intents['intents']:
            if ints['tag'] == classes[index]:
                response = random.choice(ints['responses'])
        
        return response
        
    def send_text(self):
        text = self.entry.get("1.0",'end-1c').strip()
        self.entry.delete(0.0, END)
        if text != '':
            self.chatBox.config(state = NORMAL)
            self.chatBox.insert(END, "You: " + text + '\n\n')
            self.chatBox.insert(END, "Bot: " + self.get_response(text) + '\n\n')           

               
if __name__ ==  "__main__":
    GUI()
