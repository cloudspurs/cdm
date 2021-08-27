###
# DDina: Dynamic guess and slip dina
# dina guess and slip change with time
# guess = a*arctan(b*t)
# slip = c*arccot(d*T)
#
# 先按时间划分数据
# 每轮迭代先求guess，slip，再求guess,slip与时间的参数(a, b, c, d)
# 结果：效果很差，无法使用，迭代开始时，估计的guess和slip不准，导致abcd估计也不准，
# 这样一致迭代下去，得不到准确结果
###


import math
import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from dina import Dina
from d1_dina import arccot, guess_func, slip_func


class EMt_Dina(Dina):
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

		a, b, c, d = self.a.copy(), self.b.copy(), self.c.copy(), self.d.copy()

		for i in range(epoch):
			tmp_a, tmp_b, tmp_c, tmp_d = a.copy(), b.copy(), c.copy(), d.copy()

			guess = guess_func(self.T, a, b)
			slip = slip_func(self.T, c, d)

			guess = np.where(guess > 1, 1, guess)
			guess = np.where(guess < 0, 0, guess)
			slip = np.where(slip > 1, 1, slip)
			slip = np.where(slip < 0, 0, slip)

			# 每个时间段每个题目的guess和slip, 平均时间，作答数量
			all_guess, all_slip = np.zeros((split_num, self.prob_num)), np.zeros((split_num, self.prob_num))
			answer_number, mean_time = np.zeros((split_num, self.prob_num)), np.zeros((split_num, self.prob_num))
			
			for k in range(split_num): # 每个时间段数据
				tmp_R = np.copy(self.R)

				like = np.zeros(shape=(self.stu_num, self.state_num))  # 学生在每种属性的似然 
				post = np.zeros(shape=(self.stu_num, self.state_num))  # 学生在每种属性的后验概率 

				for s in range(self.state_num):
					answer_right = guess * (1 - self.eta[s]) + (1 - slip) * self.eta[s]

					log_like = np.log(answer_right + 1e-9) * tmp_R + np.log(1 - answer_right + 1e-9) * (1 - tmp_R)

					tmp_log_like = np.where(self.T <= (duration * i), 0, log_like)
					tmp_log_like = np.where(self.T > (duration * (i+1)), 0, tmp_log_like)

					like[:, s] = np.exp(np.sum(tmp_log_like, axis=1))

				post = like / np.sum(like, axis=1, keepdims=True)

				now_question = np.where(self.T <= (duration * i), 0, 1)
				now_question = np.where(self.T > (duration * (i+1)), 0, now_question)

				sub_guess = np.zeros((self.prob_num,))
				sub_slip = np.zeros((self.prob_num,))
				for j in range(self.prob_num):
					real_post = post * now_question[:, j:j+1]

					# 每种属性向量的人数
					i_l = np.expand_dims(np.sum(real_post, axis=0), axis=1)

					# 每个题目eta=0和eta=1的人数
					i_jl_0, i_jl_1 = np.sum(i_l * (1 - self.eta[:, j:j+1]), axis=0), np.sum(i_l * self.eta[:, j:j+1], axis=0)

					# 答对人人数
					r_jl = np.dot(np.transpose(real_post), tmp_R[:, j:j+1])
					r_jl_0, r_jl_1 = np.sum(r_jl * (1 - self.eta[:, j:j+1]), axis=0), np.sum(r_jl * self.eta[:, j:j+1], axis=0)

					one_guess, one_slip = r_jl_0 / i_jl_0, (i_jl_1 - r_jl_1) / i_jl_1
					sub_guess[j], sub_slip[j] = one_guess, one_slip

				all_guess[k] = sub_guess
				all_slip[k] = sub_slip 

			for k in range(self.prob_num):
				popt, pcov = curve_fit(guess_func, mean_time[:, k], all_guess[:, k])
				a[i] = popt[0]
				b[i] = popt[1]

				popt, pcov = curve_fit(slip_func, mean_time[:, k], all_slip[:, k])
				c[i] = popt[0]
				d[i] = popt[1]

			#change = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(guess - guess_tmp)), \
			#		np.max(np.abs(slip - slip_tmp)))

			#if iteration > 20 and change < epsilon:
			#	break

		self.all_guess, self.all_slip = all_guess, all_slip

		#print(answer_number)
		#print(mean_time)

		# 计算post
		#post = np.sum(all_post, axis=0)
		#post = post / np.sum(post, axis=1, keepdims=True)
			
		# 获取最可能的属性组合
		#theta = np.argmax(post, axis=1)

		# 计算属性掌握概率
		#self.know_prob = np.dot(post, self.all_states)

		# 3. 对每个题目的guess和slip与时间的参数进行估计
		#self.theta = theta
		self.a, self.b, self.c, self.d = a, b, c, d


if __name__ == '__main__':
	q = np.loadtxt('../data/q.csv', dtype=int, delimiter=',') 		# 读取Q矩阵
	d = np.loadtxt('../data/correct_wrong.csv', dtype=int, delimiter=',')	# 读取作答对错
	rt = np.loadtxt('../data/time.csv', delimiter=',')	# 读取作答时间

	#max_num = 5000000
	#d =d[:max_num]
	#rt = rt[:max_num]

	emtdina = EMt_Dina(q, d, rt)
	emtdina.train(10, 0.001)
	print(emtdina.a)
	print(emtdina.b)
	print(emtdina.c)
	print(emtdina.d)
	for i, theta in enumerate(emtdina.theta[:10]):
		print(emtdina.all_states[theta])

	#params = np.loadtxt('../data/1000wan_time_parameters.csv', delimiter=',')
	#a = params[0]
	#b = params[1]
	#c = params[2]
	#d = params[3]

	#for k in range(emtdina.prob_num):
	#	x = np.linspace(0, 20, 100)
	#	y = a[k] * np.arctan(b[k] * x)
	#	plt.plot(x, y, label='guess true')

	#	x = range(20)
	#	plt.plot(x, emtdina.all_guess[:,k], label='guess estimated')

	#	y = c[k] * arccot(d[k] * x)
	#	plt.plot(x, y, label='slip true')

	#	x = range(20)
	#	plt.plot(x, emtdina.all_slip[:,k], label='slip estimated')

	#	plt.legend(loc='best')

	#	plt.savefig('../data/img/1000wan_' + str(k) + '.png')
	#	plt.clf()

	#estimated_params = np.array([emtdina.a, emtdina.b, emtdina.c, emtdina.d])
	#np.savetxt('../data/1000wan_estimated_time_parameters.csv', estimated_params, delimiter=',', fmt='%.5f')
	#np.savetxt('../data/1000wan_abcd_error.csv', np.abs(params - estimated_params), delimiter=',', fmt='%.5f')
	#
	#guess_error = []
	#slip_error = []
	#for i in range(1, 21):
	#	guess_error.append(np.abs(a * np.arctan(b * i) - emtdina.a * np.arctan(emtdina.b * i)))
	#	slip_error.append(np.abs(c * np.arctan(d * i) - emtdina.c * np.arctan(emtdina.d * i)))
	#
	#np.savetxt('../data/1000wan_guess_error.csv', guess_error, delimiter=',', fmt='%.5f')
	#np.savetxt('../data/1000wan_slip_error.csv', slip_error, delimiter=',', fmt='%.5f')

