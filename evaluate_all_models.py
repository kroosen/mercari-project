import os
from keras.models import load_model
import pickle
from Mercari_notebook import evaluate, rmsle_cust

# Load models
fp = 'Checkpoints/'
ext = '.h5'
models = [load_model(fp + filename, custom_objects={'rmsle_cust': rmsle_cust}) for filename in os.listdir(fp) if os.path.splitext(filename)[1] == ext]

# Load a dataset to evaluate models on
assert os.path.isfile('Dataset/dataset.pickle')
with open('Dataset/dataset.pickle', 'rb') as handle:
    dataset = pickle.load(handle)

#Evaluate the models
for m in models:
    print(m.name, end='')
    evaluate(m, dataset)
