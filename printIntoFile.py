import pandas as pd
from  sklearn.metrics import *
from settings import *

def printIntoFileResults(classificationResults,nameOfFile,nameOfOptimizer):
    df = pd.DataFrame(classificationResults).transpose()
    df.to_csv(trained_folder+"\\"+nameOfFile+"\\"+nameOfOptimizer+"\\classficationResults.csv", index=True)

def printModelIntoFile(model,nameOfFile,nameOfOptimizer):
    path = os.path.join(trained_folder, nameOfFile,nameOfOptimizer ,'Model.h5')
    model.save(path, include_optimizer=False)
