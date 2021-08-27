import numpy as np

from dina import Dina
from cdina import CDina
from tdina import TDina
from xdina import XDina
from ddina import DDina
from mdina import MDina
from d1_dina import D1_Dina


np.set_printoptions(formatter={'float': '{:0.3f}'.format}, precision=3, suppress=True)
np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
	q = np.loadtxt('../data/q.csv', dtype=int, delimiter=',') 		# 读取Q矩阵
	d = np.loadtxt('../data/correct_wrong.csv', dtype=int, delimiter=',')	# 读取作答对错
	rt = np.loadtxt('../data/time.csv', delimiter=',')	# 读取作答时间

	# D1_Dina
	d1_dina = D1_Dina(q, d, rt)
	d1_dina.train(100, 0.001)
	print(d1_dina.guess_a, d1_dina.guess_b)
	print(d1_dina.slip_a, d1_dina.slip_b)
	for i, theta in enumerate(d1_dina.theta[:10]):
		print(d1_dina.all_states[theta])

	#M-Dina
	#mdina = MDina(q, d, rt)
	#mdina.train(100, 0.001)
	#print('theta', mdina.theta[:10])
	#print('guess_slope', mdina.guess_slope)
	#print('guess_intercept', mdina.guess_intercept)
	#print('slip_slope', mdina.slip_slope)
	#print('slip_intercept', mdina.slip_intercept)

	#D-Dina
	#ddina = DDina(q, d, rt)
	#ddina.train(100, 0.001)
	#print('theta', ddina.theta[:10])
	#print('guess_slope', ddina.guess_slope)
	#print('guess_intercept', ddina.guess_intercept)
	#print('slip_slope', ddina.slip_slope)
	#print('slip_intercept', ddina.slip_intercept)

	# XDina
	#xdina = XDina(q, d, rt)
	#xdina.train(1000, 0.001)
	#print('guess', xdina.guess)
	#print('slip', xdina.slip)

	#print('eta0', xdina.miu_zero, np.sqrt(xdina.sigma_zero))
	#print('eta1', xdina.miu_one, np.sqrt(xdina.sigma_one))

	#print('theta')
	#for i, theta in enumerate(xdina.theta[30:50]):
	#	print(xdina.all_states[theta])

	# Dina模型
	#dina = Dina(q, d)
	#dina.train(1000, 0.001)
	#print('guess', dina.guess)
	#print('slip', dina.slip)
	#print('theta')
	#for i, theta in enumerate(dina.theta[30:50]):
	#	print(dina.all_states[theta])

	# C-Dina模型
	#cdina = CDina(q, rt)
	#cdina.train(1000, 0.001)
	#print('eta0', cdina.miu_zero, np.sqrt(cdina.sigma_zero))
	#print('eta1', cdina.miu_one, np.sqrt(cdina.sigma_one))
	#print('theta')
	#for i, theta in enumerate(cdina.theta[:10]):
	#	print(cdina.all_states[theta])

	# T-Dina模型
	#tdina = TDina(q, d, rt, cdina.miu_zero, cdina.sigma_zero, cdina.miu_one, cdina.sigma_one)
	#tdina.train(1000, 0.001)
	#print('guess', tdina.guess)
	#print('slip', tdina.slip)
	#print('theta')
	#for i, theta in enumerate(tdina.theta[:10]):
	#	print(tdina.all_states[theta])

