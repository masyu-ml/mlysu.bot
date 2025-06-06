import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf

model = tf.keras.models.load_model('masyu.keras')

lemmatizer = WordNetLemmatizer()
feedbacks = json.loads(open('feedbacks.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words =[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
     for i, word in enumerate(words):
         if word == w:
             bag[i] = 1
    return(np.array(bag))

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'feedback': classes[r[0]], 'probability': r[1]})
    return return_list

def model_feedbacks(feedbacks_list, feedbacks_json):
    tag = feedbacks_list[0]['feedback']
    list_of_feedbacks = feedbacks_json['feedbacks']
    for i in list_of_feedbacks:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        else:
            result = (i[''])
    return result

print("MLsyu logged in!")
print("Welcome to the MLsyu bot!")

while True:
   message = input("")
   ints = predict_class(message)
   res = model_feedbacks(ints, feedbacks)
   print(res)