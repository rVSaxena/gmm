import numpy as np
import logging
from gmm import GMM

def save_gmm(model, file):

	# Zeroth line contains num_components, dimensionality

	# First line will contain the mixing
	# coeffs, comma separated

	# The next num_components lines will contain the 1 mean array (as a comma separated str) per line
	# The next num_components lines will contain 1 flattened cov mat (as a comma separated str) per line

	logging.debug("Saving GMM to file: {}".format(file))

	def log(arr):
		f.write(",".join(map(str, arr))+"\n")

	with open(file, 'w') as f:
		
		log([model.num_components, model.dimensionality])
		log(model.mixing_coeffs)
		
		for i in range(model.num_components):
			log(model.means[i])

		for i in range(model.num_components):
			log(model.covariances[i].flatten(order='C')) # flattening by row major, ie, row1, row2, ..., last row

	return

def load_gmm(file):

	logging.debug("Loading GMM from file: {}".format(file))

	with open(file, 'r') as f:

		num_components, dimensionality=map(int, f.readline().strip().split(","))

		mixing_coeffs=np.array(list(map(int, f.readline().strip().split(","))))

		means=[]
		for i in range(num_components):
			means.append(
				list(map(int, f.readline().strip.split(",")))
				)
		means=np.array(means)

		covariances=[]
		for i in range(num_components):
			covariances.append(
				np.reshape(
					list(map(int, f.readline().strip().split(","))),
					(dimensionality, dimensionality)
					)
				)

	model=GMM(num_components, dimensionality)
	model.mixing_coeffs=mixing_coeffs
	model.means=means
	model.covariances=np.array(covariances)

	return model







