import pandas as pd
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

def _read_data():
	data = pd.read_csv('../Data/tiktok.csv')
	encode = LabelEncoder()
	for col in data.columns:
		encode.fit(data[col])
		data[col] = encode.transform(data[col])
	data = shuffle(data)
	target = data['target']
	data = data.drop(['target'],axis = 1)
	return data,target

def model1(nodes,l_rate,d_rate,momentum):
	model=Sequential()
	model.add(Dense(nodes, input_dim=9))
	model.add(Activation('sigmoid'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	sgd = SGD(lr=l_rate,momentum=momentum, decay=d_rate, nesterov=False)
	model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['binary_accuracy'])
	return model

def model2(nodes,l_rate,d_rate,momentum):
	model=Sequential()
	model.add(Dense(nodes, input_dim=9))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	sgd = SGD(lr=l_rate,momentum=momentum, decay=d_rate, nesterov=False)
	model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['binary_accuracy'])
	return model

def value(reward,hidden,n_hidden):
	if reward==0:
		l_rate = 0.1
		n_hidden=hidden
		n_hidden=n_hidden-1
	else:
		l_rate=0.04
		n_hidden=n_hidden-1
		if n_hidden<4:
			n_hidden=3

	if reward>.8:
		epochs = 300
	else:
		epochs=100
	return l_rate,n_hidden,epochs





