import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  keras import *
import tensorflow
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras import layers
from keras import models
from  models import *
from visualize import  *
from settings import  *
from printIntoFile import *
import  keras

(X_train, y_train), (X_test, y_test)= tensorflow.keras.datasets.fashion_mnist.load_data()
print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)

print(X_train[0])

print(y_train[0])

class_labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneakers','Bag','Ankle boot']

nameOfModel='DataAugmentation'
nameofOptimizer="Adam=0.001_Dropout_after_flastten=0.5"
makeFolderIfNotExist(nameOfModel,nameofOptimizer)

visualizeTrainingData(X_train,y_train,nameOfModel,nameofOptimizer);

X_train = X_train/255
X_test = X_test/255

print(X_train.ndim)
print(X_train.shape)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2],1)) #1 jer je crno bela slika
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] , X_test.shape[2],1))




print(X_train.ndim)
print(X_train.shape)
print(X_train[0])

X_train, X_val , y_train , y_val = train_test_split(X_train,y_train , test_size = 0.2 , random_state = 2020)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

model=dataGeneratedCNN()

visualizeModelAndSave(model,nameOfModel,nameofOptimizer)

gen = ImageDataGenerator(rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

batch_Size=25

batches = gen.flow(X_train, y_train, batch_size=batch_Size)
val_batches = gen.flow(X_val, y_val, batch_size=batch_Size)









visualizeModelAndSave(model,nameOfModel,nameofOptimizer)

model.summary()

opt=tensorflow.keras.optimizers.Adam(learning_rate=0.001)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max',patience=5, restore_best_weights=True, verbose=1)
model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=opt,metrics=[keras.metrics.sparse_categorical_accuracy])


hist=model.fit_generator(batches, steps_per_epoch=X_train.shape[0]/3, epochs=50,validation_data=val_batches,validation_steps=X_val.shape[0]/3,callbacks=[early_stopping])

printModelIntoFile(model,nameOfModel,nameofOptimizer)


plotLossAndSave(hist,nameOfModel,nameofOptimizer)

plotAccuracyAndSave(hist,nameOfModel,nameofOptimizer)

y_pred = model.predict(X_test).round(2)

model.evaluate(X_test,y_test)


visualizePreditions(y_test,y_pred,X_test,nameOfModel,nameofOptimizer)
visualizeClassificationAndSave(y_test,y_pred,nameOfModel,nameofOptimizer)



cReport = classification_report (y_test,[np.argmax(label) for label in y_pred],target_names = class_labels,output_dict=True)

printIntoFileResults(cReport,nameOfModel,nameofOptimizer)



plt.show()