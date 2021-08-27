import numpy as np

from dina import Dina

if __name__ == '__main__':
	q = np.loadtxt('../data/q.csv', delimiter=',')
	d = np.loadtxt('../data/correct_wrong_dina.csv', delimiter=',')
	i, j, k = d.shape[0], q.shape[0], q.shape[1]

	dina = Dina(q, d)

	dina.train(1000, 0.001)

	miu_one = np.random.uniform(0.5, 2, (j,))
	sigma_one = np.random.uniform(0.5, 2, (j,))
	miu_zero = np.zeros((j,))
	sigma_zero = np.ones((j,))

	np.savetxt('../data/c_dina_paramaters.csv', [miu_zero, sigma_zero, miu_one, sigma_one], delimiter=',')

	result = np.zeros((i, j))

	for i,index in enumerate(dina.theta):

		theta = dina.all_states[index]
		etas = 1 - (np.sum(q, axis=1) - np.dot(q, np.transpose(theta)) > 0)
		for j, eta in enumerate(etas):
			if eta == 1:
				r = np.random.lognormal(miu_one[j], sigma_one[j], 1)
				result[i][j] = r
			if eta == 0:
				r = np.random.lognormal(0, 1, 1)
				result[i][j] = r
	
	np.savetxt('../data/time_c_dina.csv', result, delimiter=',')
		
