import os


trained_folder = os.path.join('D:\\FashionMnist\\models')
path='D:\\FashionMnist\\models'

if not os.path.exists(trained_folder):
    os.mkdir(trained_folder)

def makeFolderIfNotExist(name,nameOfOptimizer):
    newFolder=os.path.join(trained_folder,name,nameOfOptimizer)
    modelFolder=os.path.join(trained_folder,name)
    if not os.path.exists(modelFolder):
        os.mkdir(modelFolder)
    if not os.path.exists(newFolder):
        os.mkdir(newFolder)

image_size = 228
batch_size = 512