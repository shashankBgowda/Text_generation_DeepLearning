
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts([dataset])
tokenizer.word_index
input_sequences=[]
for sentence in dataset.split('\n'):
    tokenized_sentence=tokenizer.texts_to_sequences([sentence])[0]
    #converted sentences to numbers , as above

    #arrenged in sequence order
    for i in range(1,len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:i+1])
input_sequences
#getting the length of all the sentences and adding 0's in fron of it
#  sentences which are lesser then max_lenth sentences
#this is called PADDING


max_len=max([len(x) for x in input_sequences])
max_len
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_input_sequence=pad_sequences(input_sequences,maxlen=max_len,padding='pre')
padded_input_sequence
X = padded_input_sequence[:,:-1]
Y=padded_input_sequence[:,-1]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
model = Sequential()
model.add(Embedding(283, 100, input_length=56))
model.add(LSTM(150))
model.add(Dense(283, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,epochs=100)
import numpy as np
import time
text = 'make the payment'

for i in range(10):
  # tokenize
  token_text = tokenizer.texts_to_sequences([text])[0]
  # padding
  padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
  # predict
  pos = np.argmax(model.predict(padded_token_text))

  for word,index in tokenizer.word_index.items():
    if index == pos:
      text = text + " " + word
      print('Loading..!')
      time.sleep(1)
      print(text)





