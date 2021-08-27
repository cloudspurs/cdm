from cdm import Cdm

import numpy as np
from lognorm import log_norm


class C_Dina(Cdm):
	# R: 作答时间矩阵
	def __init__(self, Q, R):
		super().__init__(Q, R)
		self.theta = np.zeros(shape=self.stu_num)
		self.miu_zero = np.zeros(shape=self.prob_num) + 0
		self.sigma_zero = np.zeros(shape=self.prob_num) + 1
		self.miu_one = np.zeros(shape=self.prob_num) + 1 
		self.sigma_one = np.zeros(shape=self.prob_num) + 1

	def train(self, epoch, epsilon):
		like = np.zeros(shape=(self.stu_num, self.state_num)) # 似然
		post = np.zeros(shape=(self.stu_num, self.state_num)) # 后验概率

		theta, miu_zero, sigma_zero, miu_one, sigma_one, tmp_R = np.copy(self.theta), np.copy(self.miu_zero), \
				np.copy(self.sigma_zero),np.copy(self.miu_one), np.copy(self.sigma_one), np.copy(self.R)

		for iteration in range(epoch):
			post_tmp, miu_zero_tmp, sigma_zero_tmp, miu_one_tmp, sigma_one_tmp = np.copy(post), np.copy(miu_zero), \
					np.copy(sigma_zero + 1e-9), np.copy(miu_one), np.copy(sigma_one + 1e-9)

			for s in range(self.state_num):
				like_zero = np.log(log_norm(miu_zero, sigma_zero, self.R + 1e-9) + 1e-9)
				like_one = np.log(log_norm(miu_one, sigma_one, self.R + 1e-9) + 1e-9)
				log_like = like_zero * (1 - self.eta[s, :]) + like_one * self.eta[s, :]
				like[:, s] = np.exp(np.sum(log_like, axis=1))

			post = like / np.sum(like, axis=1, keepdims=True)

			# 作答时间处于eta=0分布和eta=1分布的概率
			eta_0 = np.dot(post, 1- self.eta)
			eta_1 = np.dot(post, self.eta)
			p_eta_0 = eta_0 / np.sum(eta_0, axis=0, keepdims=True)
			p_eta_1 = eta_1 / np.sum(eta_1, axis=0, keepdims=True)

			# 更新参数
			miu_zero = np.sum(p_eta_0 * np.log(tmp_R + 1e-9), axis=0)
			miu_one = np.sum(p_eta_1 * np.log(tmp_R + 1e-9), axis=0)
			sigma_zero = np.sum(p_eta_0 * (np.log(tmp_R + 1e-9)-miu_zero)**2, axis=0)
			sigma_one = np.sum(p_eta_1 * (np.log(tmp_R + 1e-9)-miu_one)**2, axis=0)

			theta = np.argmax(post, axis=1)

			change = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(miu_zero - miu_zero_tmp)), \
					np.max(np.abs(miu_one - miu_one_tmp)), np.max(np.abs(sigma_zero - sigma_zero_tmp)), \
					np.max(np.abs(sigma_one- sigma_one_tmp)))

			if iteration > 20 and change < epsilon:
				break

		self.theta, self.miu_zero, self.miu_one, self.sigma_zero, self.sigma_one = theta, miu_zero, miu_one, sigma_zero, sigma_one

		# 计算学生属性掌握概率
		self.know_prob = np.dot(post, self.all_states) 


if __name__ == '__main__':
	q = np.loadtxt('../data/q.csv', dtype=int, delimiter=',')	# 读取Q矩阵
	d = np.loadtxt('../data/time_c_dina.csv', dtype=float, delimiter=',')	# 读取作答对错

	cdina = C_Dina(q, d)
	cdina.train(1000, 0.001)

	print('eta 0 miu', cdina.miu_zero)
	print('eta 0 sigma', np.sqrt(cdina.sigma_zero))
	print('eta 1 miu', cdina.miu_one)
	print('eta 1 sigma', np.sqrt(cdina.sigma_one))
	print('theta')
	for i, theta in enumerate(cdina.theta[:10]):
		print(cdina.all_states[theta])

