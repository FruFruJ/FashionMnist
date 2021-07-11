import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  keras import *
import tensorflow
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from  models import *
from visualize import  *
from settings import  *
from printIntoFile import *
import keras
from keras.applications import vgg16
from keras.preprocessing.image import img_to_array, array_to_img


(X_train, y_train), (X_test, y_test)= tensorflow.keras.datasets.fashion_mnist.load_data()


print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)

print(X_train[0])

print(y_train[0])

class_labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneakers','Bag','Ankle boot']

nameOfModel='ModelUsingPretrainedConvNetVGG16'
nameofOptimizer="Adam=0.001_Dropout=0.2"
makeFolderIfNotExist(nameOfModel,nameofOptimizer)


visualizeTrainingData(X_train,y_train,nameOfModel,nameofOptimizer);


print(X_train.ndim)
print(X_train.shape)


conv_base = vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(48, 48, 3))





X_test=np.dstack([X_test] * 3)
X_train=np.dstack([X_train]*3)

X_train=X_train.reshape(-1,28,28,3)
X_test=X_test.reshape(-1,28,28,3)





X_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in X_train])
X_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in X_test])


X_train = X_train/255
X_test = X_test/255

print(X_train.ndim)
print(X_train.shape)
print(X_train[0])

X_train, X_val , y_train , y_val = train_test_split(X_train,y_train , test_size = 0.2 , random_state = 2020)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)


train_features = conv_base.predict(X_train, batch_size=64, verbose=1)
test_features = conv_base.predict(X_test, batch_size=64, verbose=1)
val_features = conv_base.predict(X_val, batch_size=64, verbose=1)



model=topOfVGG()

train_features_flat = np.reshape(train_features, (train_features.shape[0], 1*1*512))
test_features_flat = np.reshape(test_features, (test_features.shape[0], 1*1*512))
val_features_flat = np.reshape(val_features, (val_features.shape[0], 1*1*512))

print('flattened')
print(train_features_flat.shape)
print(test_features_flat.shape)
print(val_features_flat.shape)


visualizeModelAndSave(model,nameOfModel,nameofOptimizer)

model.summary()

opt=tensorflow.keras.optimizers.Adam(learning_rate=0.001)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max',patience=10, restore_best_weights=True, verbose=1)

#hist = model.fit_generator(train_generator,steps_per_epoch=num_train_batches,epochs=10,validation_data=val_generator,validation_steps=num_val_batches)
model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=opt,metrics=[keras.metrics.sparse_categorical_accuracy])
hist=model.fit(train_features_flat,y_train,batch_size = batch_size,verbose = 1, validation_data=(val_features_flat,y_val),epochs=100,callbacks=[early_stopping])

printModelIntoFile(model,nameOfModel,nameofOptimizer)


plotLossAndSave(hist,nameOfModel,nameofOptimizer)

plotAccuracyAndSave(hist,nameOfModel,nameofOptimizer)

y_pred = model.predict(test_features_flat).round(2)

model.evaluate(test_features_flat,y_test)


#visualizePreditions(y_test,y_pred,X_test,nameOfModel,nameofOptimizer)
visualizeClassificationAndSave(y_test,y_pred,nameOfModel,nameofOptimizer)



cReport = classification_report (y_test,[np.argmax(label) for label in y_pred],target_names = class_labels,output_dict=True)

printIntoFileResults(cReport,nameOfModel,nameofOptimizer)