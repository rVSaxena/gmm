import logging
import json
import numpy as np
from tqdm import tqdm
from os import makedirs
from os.path import join as pathjoin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from gmm import GMM
from utils import save_gmm, load_gmm, DataWrapper
from trainConstructs import getDataset, train_constructs
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

makedirs(train_constructs["logging_dir"], exist_ok=True)
logging.basicConfig(filename=pathjoin(train_constructs["logging_dir"], ".log"), level=train_constructs["logging_level"], filemode='w+')

def print_matrix(arr):
    print(arr.shape)
    print()
    for i in range(arr.shape[0]):
        print()
        for j in range(arr.shape[1]):
            print(arr[i, j], end=",")
    print()

def save_nll_plot():
    
    fig, ax=plt.subplots()
    ax.plot(nlls)
    fig.suptitle("The negative log likelihood at every iteration")
    fig.savefig(pathjoin(train_constructs["logging_dir"], "NLL.jpeg"))

    return

dataset=getDataset() # the dataset as a np array

pca=PCA(n_components=30)
pca.fit(dataset)
mod_dataset=pca.transform(dataset)

# data_wrapper=DataWrapper(dataset)
# encoded_dataset=data_wrapper.encode(dataset)

gmm=GMM(
    train_constructs["num_components"],
    mod_dataset.shape[1],
    reg_covariance_det=train_constructs["reg_covar"]
    )

# Will do KMeans first to find good initialization for means, covariances & mixing coeffs

logging.debug("Starting KMEANS")

km=KMeans(
  n_clusters=train_constructs["num_components"],
  max_iter=200
  )

km.fit(mod_dataset)

y=km.predict(mod_dataset)

for i in range(train_constructs["num_components"]):
    gmm.means[i]=np.mean(mod_dataset[y==i], axis=0)
    gmm.covariances[i]=np.cov(mod_dataset[y==i].T) # +np.diag(np.zeros((train_constructs["dimensionality"]))+train_constructs["reg_covar"])
    logging.debug("Det of cov of {} component: {}".format(i, np.linalg.det(gmm.covariances[i])))
    gmm.mixing_coeffs[i]=np.sum(y==i)/len(y)

logging.debug("Ending KMEANS")

nlls=[]

logging.debug("Starting EM")

for opt_idx in tqdm(range(train_constructs["num_iters"])):

    logging.debug("Starting iteration {}".format(opt_idx))

    # E step

    prob, part_prob=gmm.pdf(mod_dataset)
    
    responsibilities=part_prob/prob[:, None]

    # M step

    for i in range(gmm.num_components):
        logging.debug("M step for {} component".format(i))
        gmm.means[i]=np.average(mod_dataset, axis=0, weights=responsibilities[:, i])
        gmm.covariances[i]=np.cov(mod_dataset.T, aweights=responsibilities[:, i]) # +np.diag(np.zeros((train_constructs["dimensionality"]))+train_constructs["reg_covar"])
        logging.debug("det of component {} cov: {}".format(i, np.linalg.det(gmm.covariances[i])))
        gmm.mixing_coeffs[i]=sum(responsibilities[:, i])/mod_dataset.shape[0]

    nlls.append(-np.sum(np.log(prob)))

    logging.debug("Iteration {} of EM algorithm done".format(opt_idx))

logging.debug("Ending EM")

save_gmm(gmm, pathjoin(train_constructs["logging_dir"], "trained_model.gmm"))
np.savetxt(pathjoin(train_constructs["logging_dir"], "NLL.csv"), nlls, delimiter=",", header="NEGATIVE_LOG_LIKELIHOOD")
save_nll_plot()

logging.debug("Generating samples")
samples=pca.inverse_transform(gmm.sample(50)).reshape((50, 28, 28))
fig, axs=plt.subplots(5, 10, figsize=(50, 10))

for i in range(50):
    axs[int(i/10), int(i%10)].imshow(samples[i])

fig.suptitle("50 generated samples")
fig.savefig(pathjoin(train_constructs["logging_dir"], "generated_samples.jpeg"))
fig.show()


