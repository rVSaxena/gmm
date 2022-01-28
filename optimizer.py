from argparse import ArgumentParser
import json
import numpy 
from gmm import GMM
from utils import save_gmm, load_gmm
from trainConstructs import getDataset, train_constructs

parser=ArgumentParser()
parser.add_argument("logging_dir", help="The directory where all the run log/files are dumped")
parser.parse_args()

gmm=GMM(
	train_constructs["num_components"],
	train_constructs["dimensionality"]
	)

dataset=getDataset() # the dataset as a np array

for opt_idx in range(train_constructs["num_iters"]):
	
	neg_log_likelihood=0

	# E step

	responsibilities=[]

	for i in range(len(dataset)):
		prob, part_prob=gmm.pdf(dataset[i])
		responsibilities.append(part_prob/prob)
		neg_log_likelihood+=-np.log(prob)

	responsibilities=np.array(responsibilities)

	# M step
	new_means=[]
	new_covs=[]

	for i in range(gmm.num_components):
		_N_k=sum(responsibilities[:, i])
		new_means.append(
			np.average()
			)