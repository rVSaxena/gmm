import logging
import torch
import torchaudio

train_constructs={}
train_constructs["num_components"]=1000
train_constructs["dimensionality"]=1
train_constructs["num_iters"]=1000
train_constructs["logging_dir"]="D:\\Vaibhav_Personal\\Entertainment\\song_generation\\run1_RB\\"
train_constructs["logging_level"]=20


def getDataset():
	logging.debug("Loading RB ABUDHABI dataset")
	return (torchaudio.load("data/RB_ABUDHABI.wav")[0]).numpy().reshape((-1, 1))