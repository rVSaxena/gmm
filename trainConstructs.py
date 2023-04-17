import pandas as pd
import numpy as np
import logging
from os.path import join as pathjoin

train_constructs={}
train_constructs["num_components"]=3
train_constructs["dimensionality"]=28*28
train_constructs["num_iters"]=1000
train_constructs["logging_dir"]=""
train_constructs["logging_level"]=0
train_constructs["reg_covar"]=1e-5


def getDataset():
	logging.debug("Loading MNIST train")
	data=pd.read_csv("data/mnist_train.csv").values[:100, 1:]*50/255
	print(np.linalg.det(np.cov(data.T)))
	return data