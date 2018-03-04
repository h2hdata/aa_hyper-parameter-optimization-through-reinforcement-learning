import pandas as pd
import numpy as np

import keras
from keras.callbacks import LearningRateScheduler,EarlyStopping, ModelCheckpoint
from keras import backend as K
import time
import sys

import helper


def _model(X_train, y_train,X_test, y_test):

	class h_loss(keras.callbacks.Callback):
		def on_train_begin(self, logs={}):
			self.losses = [1,1]
		def on_epoch_end(self, batch, logs={}):
			self.losses.append(logs.get('loss'))


	def step_decay(losses):
		if float(2*np.sqrt(np.array(history.losses[-1])))<derivative:
			learning_rate=0.001
			momentum=0.07
			d_rate=0.0
			return learning_rate
		else:
			learning_rate=l_rate
			return learning_rate


	reward=0
	hidden=8	
	n_hidden = 0
	derivative=.25 
	threshold=.8 
	acc=0.6 
	acc_list=[0,0]
	d_rate = 5e-6
	momentum = 0.9
	batch=90

	while acc-acc_list[-2]>0.0001:    
		
		if reward==0:
			def model(nodes,l_rate,d_rate,momentum):
				return helper.model1(nodes,l_rate,d_rate,momentum)
			hidden_layers=2            

		else:
			def model(nodes,l_rate,d_rate,momentum):
				return helper.model2(nodes,l_rate,d_rate,momentum)
			hidden_layers=1            

		history=h_loss()
		learning_rate=LearningRateScheduler(step_decay)

		l_rate,n_hidden,epochs = helper.value(reward,hidden,n_hidden)


		model=model(n_hidden,l_rate,d_rate,momentum)
		model.fit(X_train, y_train,nb_epoch=epochs,batch_size=batch,callbacks=[history,learning_rate])
		d=model.evaluate(X_train, y_train, batch_size=batch)
		reward=1-d[-1]
		y_pred=model.predict(X_test, batch_size=batch)
		acc=1-np.mean(abs(np.array([float(i) for i in y_pred])-y_test))
		acc_list.append(acc)

	model.summary()