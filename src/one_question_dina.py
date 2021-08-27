# 每次只取一个题目一个时间段数据进行参数估计
# 得到每个题目每段时间的guess和slip
# 再估计guess，slip和时间的参数(a, b, c, d)


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from dina import Dina
from d1_dina import arccot, guess_func, slip_func

if __name__ == '__main__':
	q = np.loadtxt('../data/q.csv', dtype=int, delimiter=',') 				# 读取Q矩阵
	r = np.loadtxt('../data/correct_wrong.csv', dtype=int, delimiter=',')	# 读取作答对错
	t = np.loadtxt('../data/time.csv', delimiter=',')						# 读取作答时间

	params = np.loadtxt('../data/abcd.csv', delimiter=',')
	guess_a = params[0]
	guess_b = params[1]
	slip_a = params[2]
	slip_b = params[3]
	
	ass, bs, cs, ds = [], [], [], []
	# 每个题计算guess和slip
	for i in range(q.shape[0]):
		nq = q[i:i+1, :]
		nr = r[:, i]
		nt = t[:, i]

		guess = []
		slip = []
		ave_time = []
		# 每个时间段的数据计算guess和slip
		for j in range(20):
			time = []
			data = []
			for k, nnt in enumerate(nt):
				if j < nnt <= j+1:
					data.append(nr[k])
					time.append(nnt)
			data = np.array(data)
			data = np.expand_dims(data, axis=1)

			dina = Dina(nq, data, 0.1, 0.1)
			dina.train(10, 0.001)
			guess.append(dina.guess[0])
			slip.append(dina.slip[0])
			ave_time.append(np.mean(np.array(time)))

		#print(guess)
		#print(slip)

		x = np.linspace(0, 20, 100)
		y = guess_a[i] * np.arctan(guess_b[i] * x)
		plt.plot(x, y, label='guess true')

		plt.plot(ave_time, guess, label='guess estimated')

		x = np.linspace(0, 20, 100)
		y = slip_a[i] * arccot(slip_b[i] * x)
		plt.plot(x, y, label='slip true')

		plt.plot(ave_time, slip, label='slip estimated')

		plt.legend(loc='best')

		plt.savefig('../data/img/' + str(i) + '.png')
		plt.clf()

		popt, pcov = curve_fit(guess_func, ave_time, guess)
		a = popt[0]
		b = popt[1]

		popt, pcov = curve_fit(slip_func, ave_time, slip)
		c = popt[0]
		d = popt[1]

		ass.append(a)
		bs.append(b)
		cs.append(c)
		ds.append(d)


	estimated_params = np.array([ass, bs, cs, ds])

	np.savetxt('../data/one_estimated_abcd.csv', estimated_params, delimiter=',', fmt='%.5f')
	
	np.savetxt('../data/one_abcd_error.csv', np.abs(params - estimated_params), delimiter=',', fmt='%.5f')
	
	guess_error = []
	slip_error = []
	for i in range(1, 21):
		guess_error.append(np.abs(guess_a * np.arctan(guess_b * i) - a * np.arctan(b * i)))
		slip_error.append(np.abs(slip_a * np.arctan(slip_b * i) - c * np.arctan(d * i)))
	
	np.savetxt('../data/one_guess_error.csv', guess_error, delimiter=',', fmt='%.5f')
	np.savetxt('../data/one_slip_error.csv', slip_error, delimiter=',', fmt='%.5f')

