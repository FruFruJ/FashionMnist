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
from keras.preprocessing.image import img_to_array, array_to_img
from keras.applications import vgg19
from keras.applications.vgg19 import preprocess_input

(X_train, y_train), (X_test, y_test)= tensorflow.keras.datasets.fashion_mnist.load_data()


print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)

print(X_train[0])

print(y_train[0])

class_labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneakers','Bag','Ankle boot']

nameOfModel='ModelUsingPretrainedConvNet'
makeFolderIfNotExist(nameOfModel)

visualizeTrainingData(X_train,y_train,nameOfModel);

print(X_train.ndim)
print(X_train.shape)



#X_train  = np.expand_dims(X_train,-1)
#X_test  = np.expand_dims(X_test,-1)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

#X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2],3)) #1 jer je crno bela slika
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] , X_test.shape[2],3))

#X_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((150,150))) for im in X_test])
#X_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((150,150))) for im in X_test])


X_train = X_train/255
X_test = X_test/255

print(X_train.ndim)
print(X_train.shape)
print(X_train[0])

X_train, X_val , y_train , y_val = train_test_split(X_train,y_train , test_size = 0.2 , random_state = 2020)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)





model=dataGeneratedCNN()

#gen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
#batches = gen.flow(X_train, y_train, batch_size=3)
#val_batches = gen.flow(X_val, y_val, batch_size=3)








#tensorflow.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
visualizeModelAndSave(model,nameOfModel)

model.summary()

#model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer='adam',metrics=[keras.metrics.sparse_categorical_accuracy])
model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(monitor='accuracy', mode='max',patience=5, restore_best_weights=True, verbose=1)

#hist = model.fit_generator(train_generator,steps_per_epoch=num_train_batches,epochs=10,validation_data=val_generator,validation_steps=num_val_batches)
hist=model.fit(X_train,y_train,batch_size = batch_size,verbose = 1, validation_data=(X_val,y_val),epochs=100,callbacks=[early_stopping])

#hist=model.fit_generator(batches, steps_per_epoch=X_train.shape[0], epochs=7,validation_data=val_batches, validation_steps=10)

printModelIntoFile(model,nameOfModel)

plotLossAndSave(hist,nameOfModel)

plotAccuracyAndSave(hist,nameOfModel)

y_pred = model.predict(X_test).round(2)

model.evaluate(X_test,y_test)


visualizeClassificationAndSave(y_test,y_pred,nameOfModel)

#visualizePreditions(y_test,y_pred,X_test)


cReport = classification_report (y_test,[np.argmax(label) for label in y_pred],target_names = class_labels,output_dict=True)

printIntoFileResults(cReport,nameOfModel)




plt.show()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
