import os
import numpy as np
import pandas as pd
import lasagne
from lasagne import layers
from lasagne.updates import adadelta
from nolearn.lasagne import NeuralNet


train = os.getcwd() + '\\train.csv'
ftest = os.getcwd() + '\\test.csv'

def load(test=False):
    name = ftest if test else train
    df = pd.read_csv(name)
    
    if not test:
        X = np.array(df.iloc[:,1:].values)
        X = np.array(X).reshape((-1, 1, 28, 28)).astype(np.uint8)
        y = df[[0]].values.ravel()
        y = y.astype(np.uint8)
        print(X.shape)
    else:
        X = np.array(df.values)
        X = X.reshape(-1, 1, 28, 28).astype(np.uint8)
        y = None
    return X,y


def CNN(m_epochs):  
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 1, 28, 28),
        conv1_num_filters=32,
        conv1_filter_size=(3, 3),
        conv1_nonlinearity=lasagne.nonlinearities.rectify,
        pool1_pool_size=(2, 2),
        dropout1_p = 0.1,
        
        conv2_num_filters=64,
        conv2_filter_size=(2, 2),
        conv2_nonlinearity=lasagne.nonlinearities.rectify,
        pool2_pool_size=(2, 2),
        dropout2_p = 0.2,
    
        conv3_num_filters=128,
        conv3_filter_size=(2, 2),
        conv3_nonlinearity=lasagne.nonlinearities.rectify,
        pool3_pool_size=(2, 2),
        dropout3_p = 0.3,
        
        hidden4_num_units=1000,
        dropout4_p = 0.5,
        hidden5_num_units=1000,
        output_num_units=10,
        output_nonlinearity=lasagne.nonlinearities.softmax,
    
#        update_learning_rate=0.001,
#        update_momentum=0.9,
        update = adadelta,
        
        regression = False,
        max_epochs=m_epochs,
        verbose=1,
        )
    return net

X, y = load()    
cnn = CNN(1500).fit(X,y)
X_test, _ = load(test=True)
predict = cnn.predict(X_test)
np.savetxt('submission_cnn.csv', np.c_[range(1,len(X_test)+1),predict],
            delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
