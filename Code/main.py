import helper
import model


data,target = helper._read_data()
l = len(data)
X_train = data[:int(l*0.6)]
X_test = data[int(l*0.6):]
y_train = target[:int(l*0.6)]
y_test = target[int(l*0.6):]
model._model(X_train, y_train,X_test, y_test)
