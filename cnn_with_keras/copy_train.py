import shutil
import os
source='C:/Users/hp/Downloads/Compressed/mnistasjpg/trainingSet/trainingSet/'
destination='C:/Users/hp/machine_learning/cnn_with_keras/train_data/'
source_folders=os.listdir(source)
for folder in source_folders:
    images=os.listdir(source+folder)[:4000]
    for image in images:
        shutil.copy(source+folder+'/'+image,destination+folder)
    