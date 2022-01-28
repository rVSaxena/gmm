import logging
import json
import numpy
import tqdm
from os import makedirs
from os.path import join as pathjoin
from gmm import GMM
from utils import save_gmm, load_gmm
from trainConstructs import getDataset, train_constructs
from matplotlob import pyplot as plt

logging.basicConfig(filename=pathjoin(train_constructs["logging_dir"], ".log"), encoding='utf-8', level=train_constructs["logging_level"])

def save_nll_plot():
	
	fig, ax=plt.subplots()
	ax.plot(nlls)
	fig.suptitle("The negative log likelihood at every iteration")
	fig.savefig(pathjoin(train_constructs["logging_dir"], "NLL.jpeg"))

	return

gmm=GMM(
	train_constructs["num_components"],
	train_constructs["dimensionality"]
	)

dataset=getDataset() # the dataset as a np array
nlls=[]

logging.debug("Starting EM")

for opt_idx in tqdm(range(train_constructs["num_iters"])):

	# E step

	prob, part_prob=gmm.pdf(dataset)
	responsibilities=part_prob/prob

	# M step

	for i in range(gmm.num_components):
		gmm.means[i]=np.average(dataset, axis=0, weights=responsibilities[:, i])
		gmm.covariances[i]=np.cov(dataset.T, aweights=responsibilities[:, i])
		gmm.mixing_coeffs[i]=sum(responsibilities[:, i])/dataset.shape[0]

	nlls.append(-np.sum(np.log(prob)))

	logging.debug("Iteraion {} of EM algorithm done".format(opt_idx))

makedirs(train_constructs["logging_dir"], exist_ok=True)
save_gmm(gmm, pathjoin(train_constructs["logging_dir"], "trained_model.gmm"))
np.savetxt(pathjoin(train_constructs["logging_dir"], "NLL.csv"), nlls, delim=",", header="NEGATIVE_LOG_LIKELIHOOD")
save_nll_plot()