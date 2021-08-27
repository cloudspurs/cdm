import numpy as np

# miu: mean, sigma: variance
def log_norm(miu, sigma, x):
	# 防止sigma为0出现None
	sigma = np.where(sigma == 0, 1e-9, sigma)
	#x = np.where(x == 0, 1e-9, x)

	return np.exp(-(np.log(x)-miu)**2/(2*sigma)) / (x*np.sqrt(2*np.pi*sigma))
