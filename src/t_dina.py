# 先估计cdina参数，再结合dina估计guess和slip
# 计算loglike时，再dina基础上，乘以时间的概率密度函数值

import numpy as np
from lognorm import log_norm
from cdm import Cdm
from dina import Dina


class TDina(Dina):
	"""
		Q: Q矩阵
		R: 作答时间
		RT: 作答时间
	"""

	def __init__(self, Q, R, RT, miu_zero, sigma_zero, miu_one, sigma_one):

		super().__init__(Q, R)

		self.miu_zero, self.sigma_zero, self.miu_one, self.sigma_one = miu_zero, sigma_zero, miu_one, sigma_one
		self.log_rt_zero = np.ones((self.stu_num, self.prob_num))
		self.log_rt_one = np.ones((self.stu_num, self.prob_num))
		for i in range(len(self.sigma_zero)):
			self.log_rt_zero[:, i] = log_norm(miu_zero[i], sigma_zero[i], RT[:, i])
			self.log_rt_one[:, i] = log_norm(miu_one[i], sigma_one[i], RT[:, i])


	def train(self, epoch, epsilon):
		like = np.zeros(shape=(self.stu_num, self.state_num))
		post = np.zeros(shape=(self.stu_num, self.state_num))

		theta, slip, guess, tmp_R = np.copy(self.theta), np.copy(self.slip), np.copy(self.guess), np.copy(self.R)

		for iteration in range(epoch):
			post_tmp, slip_tmp, guess_tmp = np.copy(post), np.copy(slip), np.copy(guess)
			# 每种状态下题目答对概率
			answer_right = (1 - slip) * self.eta + guess * (1 - self.eta)

			# 每个状态下，每个学生在所有题目的似然
			for s in range(self.state_num):

				# dina + exponential function
				#log_like = np.log(answer_right[s, :] * 0.5**self.RT + 1e-9) * self.R + np.log((1 - answer_right[s, :]) * 0.5**self.RT + 1e-9) * (1 - self.R)

				# dina + c-dina
				log_rt = self.eta[s] * self.log_rt_one + [1 - self.eta[s]] * self.log_rt_zero
				log_like = np.log(answer_right[s, :] * log_rt + 1e-9) * self.R + np.log((1 - answer_right[s, :]) * log_rt + 1e-9) * (1 - self.R)

				# 每个状态下，每个学生所有题目的似然和
				like[:, s] = np.exp(np.sum(log_like, axis=1))

			post = like / np.sum(like, axis=1, keepdims=True) # postrior probability / 每个学生在每种状态的概率

			# 每种状态下的学生人数
			i_l = np.expand_dims(np.sum(post, axis=0), axis=1)
			# 每种状态下，eta=0 和 eta=1的学生人数
			i_jl_0, i_jl_1 = np.sum(i_l * (1 - self.eta), axis=0), np.sum(i_l * self.eta, axis=0)

			# 每种状态下每个题目的答对人数
			r_jl = np.dot(np.transpose(post), tmp_R)
			# 每个题目，eta=0答对的人数 和 eta=1答对的人数
			r_jl_0, r_jl_1 = np.sum(r_jl * (1 - self.eta), axis=0), np.sum(r_jl * self.eta, axis=0)


			guess, slip = r_jl_0 / i_jl_0, (i_jl_1 - r_jl_1) / i_jl_1

			change = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(slip - slip_tmp)), np.max(np.abs(guess - guess_tmp)))
			theta = np.argmax(post, axis=1)
			if iteration > 20 and change < epsilon:
				break

		self.theta, self.slip, self.guess = theta, slip, guess
		# 学生属性掌握概率
		self.know_prob = np.dot(post, self.all_states)

