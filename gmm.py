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
		part_prob=[
		(
			self.mixing_coeffs[i]*np.exp(-0.5*(x-self.means[i]).T@np.linalg.inv(self.covariances[i])@(x-self.means[i]))
			/
			np.sqrt(((2*np.pi)**dimensionality)*np.linalg.det(self.covariances[i]))
			)

		for i in range(self.num_components)
		]

		return sum(part_prob), part_prob

	def responsibilities(self, x):

		a, b=self.pdf(x)
		return b/a

	