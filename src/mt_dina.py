import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from dina import Dina
from d1_dina import arccot, guess_func, slip_func


class Mt_Dina(Dina):
	def __init__(self, Q, R, T):
		super().__init__(Q, R)

		self.T = T
		print('max time:', np.max(T, axis=0))
		print('min time:', np.min(T, axis=0))

		self.theta = np.zeros(shape=self.stu_num)
		self.a = np.zeros(shape=self.prob_num)
		self.b = np.zeros(shape=self.prob_num)
		self.c = np.zeros(shape=self.prob_num)
		self.d = np.zeros(shape=self.prob_num)


	def train(self, epoch, epsilon, duration=1, max_time=20): # duration: 每个时间段长度
		# 计算时间段数量
		split_num = math.ceil(max_time / duration)
		print('Split num:', split_num)

		# 每个时间段每个题目的guess和slip, 平均时间，作答数量
		all_guess, all_slip = np.zeros((split_num, self.prob_num)), np.zeros((split_num, self.prob_num))
		answer_number, mean_time = np.zeros((split_num, self.prob_num)), np.zeros((split_num, self.prob_num))

		like = np.zeros(shape=(self.stu_num, self.state_num))  # 学生在每种属性向量的似然 
		post = np.zeros(shape=(self.stu_num, self.state_num))  # 学生在每种属性向量的后验概率 

		guess, slip = np.zeros((self.stu_num, self.prob_num)), np.zeros((self.stu_num, self.prob_num))

		for e in range(epoch):
			post_tmp, all_guess_tmp, all_slip_tmp = post.copy(), all_guess.copy(), all_slip.copy()

			# 根据时间获取guess和slip(每个时间段数据有不同的guess和slip)
			for i in range(self.stu_num):
				for j in range(self.prob_num):
					guess[i][j] = all_guess[self.T[i][j]-1][j]
					slip[i][j] = all_slip[self.T[i][j]-1][j]
		
			for s in range(self.state_num):
				answer_right = guess * (1 - self.eta[s]) + (1 - slip) * self.eta[s]
				log_like = np.log(answer_right + 1e-9) * self.R + np.log(1 - answer_right + 1e-9) * (1 - self.R)
				like[:, s] = np.exp(np.sum(log_like, axis=1))
			post = like / np.sum(like, axis=1, keepdims=True)

			for i in range(split_num):
				if e == 0:
					# 计算平均时间
					tmp_T = np.where(self.T <= (duration * i), 0, self.T)
					tmp_T = np.where(self.T > (duration * (i+1)), 0, tmp_T)
					no_zero_number = np.where(self.T <= (duration * i), 0, 1)
					no_zero_number = np.where(self.T > (duration * (i+1)), 0, no_zero_number)
					answer_number[i] = np.sum(no_zero_number, axis=0) # 当前时间段内每个题目作答数量
					mean_time[i] = np.sum(tmp_T, axis=0) / np.sum(no_zero_number, axis=0)

				index = np.where(self.T <= (duration * i), 0, 1)
				index = np.where(self.T > (duration * (i+1)), 0, index)
				for j in range(self.prob_num):
					real_post = post * index[:, j:j+1]	

					i_l = np.expand_dims(np.sum(real_post, axis=0), axis=1) # 每种属性向量的人数
					# 每个题目eta=0和eta=1的人数
					i_jl_0, i_jl_1 = np.sum(i_l * (1 - self.eta[:, j:j+1]), axis=0), np.sum(i_l * self.eta[:, j:j+1], axis=0)

					r_jl = np.dot(np.transpose(real_post), self.R[:, j:j+1]) # 答对人人数
					r_jl_0, r_jl_1 = np.sum(r_jl * (1 - self.eta[:, j:j+1]), axis=0), np.sum(r_jl * self.eta[:, j:j+1], axis=0)

					all_guess[i][j], all_slip[i][j] = r_jl_0 / i_jl_0, (i_jl_1 - r_jl_1) / i_jl_1

			change = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(all_guess - all_guess_tmp)), \
					np.max(np.abs(all_slip - all_slip_tmp)))
			if e > 20 and change < epsilon:
				break

		self.all_guess, self.all_slip = all_guess, all_slip
		self.theta = np.argmax(post, axis=1) # 获取最可能的属性组合
		self.know_prob = np.dot(post, self.all_states) # 计算属性掌握概率

		# 3. 对每个题目的guess和slip与时间的参数进行估计
		a, b, c, d = np.zeros((self.prob_num,)), np.zeros((self.prob_num,)), \
				np.zeros((self.prob_num,)), np.zeros((self.prob_num,))
		for i in range(self.prob_num):
			popt, pcov = curve_fit(guess_func, mean_time[:, i], all_guess[:, i])
			a[i] = popt[0]
			b[i] = popt[1]

			popt, pcov = curve_fit(slip_func, mean_time[:, i], all_slip[:, i])
			c[i] = popt[0]
			d[i] = popt[1]

		self.a, self.b, self.c, self.d = a, b, c, d


if __name__ == '__main__':
	q = np.loadtxt('../data/q.csv', dtype=int, delimiter=',') 		# 读取Q矩阵
	d = np.loadtxt('../data/correct_wrong.csv', dtype=int, delimiter=',')	# 读取作答对错
	rt = np.loadtxt('../data/time.csv', dtype=int, delimiter=',')	# 读取作答时间

	mtdina = Mt_Dina(q, d, rt)
	mtdina.train(10, 0.001)
	print(mtdina.all_guess)
	print(mtdina.all_slip)
	print(mtdina.a)
	print(mtdina.b)
	print(mtdina.c)
	print(mtdina.d)
	for i, theta in enumerate(mtdina.theta[:10]):
		print(mtdina.all_states[theta])

	params = np.loadtxt('../data/abcd.csv', delimiter=',')
	a = params[0]
	b = params[1]
	c = params[2]
	d = params[3]

	for k in range(mtdina.prob_num):
		x = np.linspace(0, 20, 100)
		y = a[k] * np.arctan(b[k] * x)
		plt.plot(x, y, label='guess true')

		x = range(20)
		plt.plot(x, mtdina.all_guess[:,k], label='guess estimated')

		y = c[k] * arccot(d[k] * x)
		plt.plot(x, y, label='slip true')

		x = range(20)
		plt.plot(x, mtdina.all_slip[:,k], label='slip estimated')

		plt.legend(loc='best')

		plt.savefig('../data/img/' + str(k) + '.png')
		plt.clf()

	estimated_params = np.array([mtdina.a, mtdina.b, mtdina.c, mtdina.d])
	np.savetxt('../data/estimated_abcd.csv', estimated_params, delimiter=',', fmt='%.5f')
	np.savetxt('../data/abcd_error.csv', np.abs(params - estimated_params), delimiter=',', fmt='%.5f')
	
	guess_error = []
	slip_error = []
	for i in range(1, 21):
		guess_error.append(np.abs(a * np.arctan(b * i) - mtdina.a * np.arctan(mtdina.b * i)))
		slip_error.append(np.abs(c * np.arctan(d * i) - mtdina.c * np.arctan(mtdina.d * i)))
	
	np.savetxt('../data/guess_error.csv', guess_error, delimiter=',', fmt='%.5f')
	np.savetxt('../data/slip_error.csv', slip_error, delimiter=',', fmt='%.5f')

