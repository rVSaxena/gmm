import numpy as np
from numpy.random import default_rng

class GMM:

	def __init__(self, num_components, dimensionality, **kwargs):

		"""
		kwargs: can supply the regularizer for the determinant of the covariance matrix
		"""

		self.num_components=num_components
		self.dimensionality=dimensionality
		mixing_coeffs=np.random.uniform(0, 1, size=(num_components, ))
		self.mixing_coeffs=mixing_coeffs/np.sum(mixing_coeffs)
		self.means=np.random.uniform(0, 1, size=(num_components, dimensionality))
		self.covariances=np.array([np.eye(dimensionality) for _ in range(num_components)])
		self.reg_covariance_det=kwargs.get("reg_covariance_det", 1e-6)
		self.rng=default_rng()

		return


	def sample(self, n_samples=1):

		sampled_component=(self.rng.multinomial(1, self.mixing_coeffs, size=(n_samples, ))@np.linspace(0, self.num_components-1, self.num_components)).astype('int')
		raw_samples=self.rng.standard_normal(size=(n_samples, self.dimensionality))
		sampled_covariances=self.covariances[sampled_component]
		sampled_means=self.means[sampled_component]
		cholesky_sampled_covariances=np.linalg.cholesky(sampled_covariances)
		real_samples=(cholesky_sampled_covariances@raw_samples[:, :, None]).squeeze()+sampled_means

		return real_samples


	def pdf(self, x):

		dimensionality=len(self.means[0])
		x=x.reshape((-1, dimensionality))

		part_prob=np.zeros((x.shape[0], self.num_components))

		for i in range(self.num_components):
			# self.covariances[i]=np.maximum(self.covariances[i], np.eye(self.dimensionality))

			self.covariances[i]=self.covariances[i]+self.reg_covariance_det*np.eye(self.dimensionality)

			exponent=np.multiply(-0.5*(x-self.means[i]), (x-self.means[i])@np.linalg.inv(self.covariances[i]))
			exponent=exponent.sum(axis=1)
			# print("max value of exponent of component {} before normalize: {}".format(i, exponent.sum(axis=1).max()))
			# stability_threshold=exponent.max()-10
			# print("stability_threshold for {} component: {}".format(i, stability_threshold))
			# exponent=exponent-stability_threshold
			# print(exponent)

			exponential=np.exp(exponent)
			part_prob[:, i]=self.mixing_coeffs[i]*np.sqrt((2*np.pi)**(-1*dimensionality))*exponential
			part_prob[:, i]=part_prob[:, i]/np.sqrt(np.linalg.det(self.covariances[i]))

		return np.sum(part_prob, axis=1), part_prob


	def responsibilities(self, x):

		a, b=self.pdf(x)
		return b/a[:, None]

	