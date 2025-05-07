import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


feedbacks = json.loads(open('feedbacks.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '.', ',', '!']

for feedback in feedbacks['feedbacks']:
 for pattern in feedback['patterns']:
     word_list = nltk.word_tokenize(pattern)
     words.extend(word_list)
     documents.append((word_list, feedback['tag']))
     if feedback['tag'] not in classes:
         classes.append(feedback['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_array = [0]  * len(classes)
#Logic for replies
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1 if word in word_patterns else  0)

    output_row = list(output_array)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)
#Model training section
random.shuffle(training)
training = np.array(training)
train_x = training[:, :-len(classes)]
train_y = training[:, -len(classes):]
#Utilizing Tensorflow 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('masyu.keras')
print('Done')

