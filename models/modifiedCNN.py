from sklearn.model_selection import train_test_split
from  models import *
from visualize import  *
from printIntoFile import *
import  tensorflow
import keras
from sklearn import preprocessing
from  keras import regularizers


print("Num GPUs Available", len(tensorflow.config.experimental.list_physical_devices('GPU')))

(X_train, y_train), (X_test, y_test)= tensorflow.keras.datasets.fashion_mnist.load_data()


print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)

print(X_train[0])

print(y_train[0])

class_labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneakers','Bag','Ankle boot']

nameOfModel='modifiedCNN'
nameofOptimizer="Adam=0.001_Dropout=0.2"
makeFolderIfNotExist(nameOfModel,nameofOptimizer)

visualizeTrainingData(X_train,y_train,nameOfModel,nameofOptimizer);


print(X_train.ndim)
print(X_train.shape)



X_train = X_train/255
X_test = X_test/255
#np.interp(X_train, (X_train.min(), X_train.max()), (-1, +1))
#np.interp(X_test, (X_test.min(), X_test.max()), (-1, +1))

print(X_train[0])


X_train  = np.expand_dims(X_train,-1)
X_test  = np.expand_dims(X_test,-1)


print(X_train.ndim)
print(X_train.shape)
print(X_train[0])

X_train, X_val , y_train , y_val = train_test_split(X_train,y_train , test_size = 0.2 , random_state = 2020)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

#regularizer=regularizers.l2(0.01)
#model=modifiedBasicCNN()
model=modifiedBasicCNNWithDropout(0.2)

visualizeModelAndSave(model,nameOfModel,nameofOptimizer)

model.summary()


#model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#opt = keras.optimizers.Adam(learning_rate=0.001)
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.001)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max',patience=10, restore_best_weights=True, verbose=1)
model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=opt,metrics=[keras.metrics.sparse_categorical_accuracy])
hist=model.fit(X_train,y_train,batch_size = batch_size,verbose = 1, validation_data=(X_val,y_val),epochs=1000,callbacks=[early_stopping])

printModelIntoFile(model,nameOfModel,nameofOptimizer)


plotLossAndSave(hist,nameOfModel,nameofOptimizer)

plotAccuracyAndSave(hist,nameOfModel,nameofOptimizer)

y_pred = model.predict(X_test).round(2)

model.evaluate(X_test,y_test)


visualizePreditions(y_test,y_pred,X_test,nameOfModel,nameofOptimizer)
visualizeClassificationAndSave(y_test,y_pred,nameOfModel,nameofOptimizer)



cReport = classification_report (y_test,[np.argmax(label) for label in y_pred],target_names = class_labels,output_dict=True)

printIntoFileResults(cReport,nameOfModel,nameofOptimizer)