import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from  settings import *
from  keras.utils.vis_utils import plot_model


class_labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneakers','Bag','Ankle boot']


def visualizeTrainingData(X_train, y_train,nameOfFile,nameOfOptimizer):
    plt.clf()
    plt.figure(figsize=(16, 16))
    j = 1
    for i in np.random.randint(0, 1000, 25):
        plt.subplot(5, 5, j);
        j += 1
        plt.imshow(X_train[i], cmap='Greys')
        plt.axis('off')
        plt.title('{} / {}'.format(class_labels[y_train[i]], y_train[i]))
        plt.savefig(os.path.join(trained_folder, nameOfFile,nameOfOptimizer, nameOfFile + 'training_data_visualization.png'))

def visualizeClassificationAndSave(y_test,y_pred,nameOfFile,nameOfOptimizer):
    plt.clf()
    plt.figure(figsize=(16, 9))
    y_pred_labels = [np.argmax(label) for label in y_pred]
    cm = confusion_matrix(y_test, y_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.savefig(os.path.join(trained_folder,nameOfFile,nameOfOptimizer, nameOfFile + '_class_classification.png'))

def visualizePreditions(y_test,y_pred,X_test,nameOfFile,nameOfOptimizer):
    plt.clf()
    plt.figure(figsize=(16, 30))
    j = 1
    for i in np.random.randint(0, 1000, 60):
        plt.subplot(10, 6, j);
        j += 1
        plt.imshow(X_test[i].reshape(28, 28), cmap='Greys')
        plt.axis('off')
        plt.title("Actual = {} / {} \n Predicted = {} / {}".format(class_labels[y_test[i]], y_test[i],
                                                                  class_labels[np.argmax(y_pred[i])],
                                                                  np.argmax(y_pred[i])))
    plt.savefig(os.path.join(trained_folder, nameOfFile, nameOfOptimizer, nameOfFile + '_visualizedPredictions.png'))

def plotLossAndSave(model,nameOfFile,nameOfOptimizer):
    plt.clf()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(model.history['loss'],'mo',label="Training Loss")
    plt.plot(model.history['val_loss'],'c',label="Validation loss")
    plt.legend()
    plt.savefig(os.path.join(trained_folder, nameOfFile, nameOfOptimizer, nameOfFile + '_loss.png'))

def plotAccuracyAndSave(model,nameOfFile,nameOfOptimizer):
    plt.clf()
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(model.history['sparse_categorical_accuracy'],'mo',label="Training sparse_categorical_accuracy")
    plt.plot(model.history['val_sparse_categorical_accuracy'],'c',label="Validation sparse_categorical_accuracy")
    plt.legend()
    plt.savefig(os.path.join(trained_folder, nameOfFile,nameOfOptimizer,nameOfFile+'_accuracy.png'))

def visualizeModelAndSave(model,nameOfFile,nameOfOptimizer):
    plt.clf()
    path=os.path.join(trained_folder, nameOfFile,nameOfOptimizer,nameOfFile+'_model_plot.png')
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)




