import numpy as np

class GMM:

	def __init__(self, num_components, dimensionality):

		self.num_components=num_components

		mixing_coeffs=np.random.uniform(0, 1, size=(num_components, ))
		self.mixing_coeffs=mixing_coeffs/np.sum(mixing_coeffs)
		self.means=np.random.uniform(0, 1, size=(num_components, dimensionality))
		self.covariances=np.array([np.eye(dimensionality) for _ in range(num_components)])

		return

	def pdf(self, x):

		dimensionality=len(self.means[0])
		x=x.reshape((-1, dimensionality))

		part_prob=np.zeros((x.shape[0], self.num_components))

		for i in range(self.num_components):
			part_prob[:, i]=self.mixing_coeffs[i]*np.exp(
				np.sum(
					np.multiply(
						-0.5*(x-self.means[i]), (x-self.means[i])*np.linalg.inv(self.covariances[i])
					),
				axis=1
				)
			)/np.sqrt(((2*np.pi)**dimensionality)*np.linalg.det(self.covariances[i]))

		return np.sum(part_prob, axis=1), part_prob


	def responsibilities(self, x):

		a, b=self.pdf(x)
		return b/a

	