# 同时使用作答对错和时间计算loglike，乘起来后会乱套
import pickle
import numpy as np
from dina import Dina
from lognorm import log_norm

# 初始化模型参数
def init_parameters(stu_num, prob_num):
	theta = np.zeros(shape=stu_num)
	miu_zero = np.zeros(shape=prob_num) + 5
	sigma_zero = np.zeros(shape=prob_num) + 1
	miu_one = np.zeros(shape=prob_num) + 1 
	sigma_one = np.zeros(shape=prob_num) + 1
	return theta, miu_zero, sigma_zero, miu_one, sigma_one


class XDina(Dina):
	def __init__(self, Q, R, RT):
		super().__init__(Q, R)
		self.RT = RT
		self.theta, self.miu_zero, self.sigma_zero, self.miu_one, self.sigma_one = init_parameters(self.stu_num, self.prob_num)
		

	# EM算法估计模型参数
	def train(self, epoch, epsilon):
		like = np.zeros(shape=(self.stu_num, self.state_num))  # 学生在每种属性的似然 
		post = np.zeros(shape=(self.stu_num, self.state_num))  # 学生在每种属性的后验概率 

		theta, slip, guess, tmp_R = np.copy(self.theta), np.copy(self.slip), np.copy(self.guess), np.copy(self.R)
		theta, miu_zero, sigma_zero, miu_one, sigma_one, tmp_R = np.copy(self.theta), np.copy(self.miu_zero), \
				np.copy(self.sigma_zero), np.copy(self.miu_one), np.copy(self.sigma_one), np.copy(self.R)

		for iteration in range(epoch):
			post_tmp, slip_tmp, guess_tmp = np.copy(post), np.copy(slip), np.copy(guess)

			post_tmp, miu_zero_tmp, sigma_zero_tmp, miu_one_tmp, sigma_one_tmp = np.copy(post), np.copy(miu_zero), \
					np.copy(sigma_zero + 1e-5), np.copy(miu_one), np.copy(sigma_one + 1e-5)

			# 每种状态下每个题目答对概率
			answer_right = (1 - slip) * self.eta + guess * (1 - self.eta)

			# 每个状态下，每个学生在所有题目的似然
			for s in range(self.state_num):
				like_zero = np.log(log_norm(miu_zero, sigma_zero, self.R + 1e-9) + 1e-9)
				like_one = np.log(log_norm(miu_one, sigma_one, self.R + 1e-9) + 1e-9)
				log_like_cdina = like_zero * (1 - self.eta[s, :]) + like_one * self.eta[s, :]

				log_like_dina = np.log(answer_right[s, :] + 1e-9) * self.R + np.log(1 - answer_right[s, :] + 1e-9) * (1 - self.R)

				log_like = log_like_cdina + log_like_dina

				# 每个状态下，每个学生所有题目的似然和
				like[:, s] = np.exp(np.sum(log_like, axis=1))

			post = like / np.sum(like, axis=1, keepdims=True) # 学生在每种状态的后验概率

			# dina参数更新
			i_l = np.expand_dims(np.sum(post, axis=0), axis=1)
			i_jl_0, i_jl_1 = np.sum(i_l * (1 - self.eta), axis=0), np.sum(i_l * self.eta, axis=0)
			r_jl = np.dot(np.transpose(post), tmp_R)
			r_jl_0, r_jl_1 = np.sum(r_jl * (1 - self.eta), axis=0), np.sum(r_jl * self.eta, axis=0)
			guess, slip = r_jl_0 / i_jl_0, (i_jl_1 - r_jl_1) / i_jl_1


			# cdina参数更新
			eta_0 = np.dot(post, 1- self.eta)
			eta_1 = np.dot(post, self.eta)
			p_eta_0 = eta_0 / np.sum(eta_0, axis=0, keepdims=True)
			p_eta_1 = eta_1 / np.sum(eta_1, axis=0, keepdims=True)
			miu_zero = np.sum(p_eta_0 * np.log(tmp_R + 1e-9), axis=0)
			miu_one = np.sum(p_eta_1 * np.log(tmp_R + 1e-9), axis=0)
			sigma_zero = np.sum(p_eta_0 * (np.log(tmp_R + 1e-9)-miu_zero)**2, axis=0)
			sigma_one = np.sum(p_eta_1 * (np.log(tmp_R + 1e-9)-miu_one)**2, axis=0)

			# 获取最可能的属性组合
			theta = np.argmax(post, axis=1)

			change0 = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(miu_zero - miu_zero_tmp)), \
					np.max(np.abs(miu_one - miu_one_tmp)), np.max(np.abs(sigma_zero - sigma_zero_tmp)), \
					np.max(np.abs(sigma_one- sigma_one_tmp)))

			change1 = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(slip - slip_tmp)), np.max(np.abs(guess - guess_tmp)))

			change = max(change0, change1)

			if iteration > 20 and change < epsilon:
				break

		self.theta, self.slip, self.guess = theta, slip, guess
		self.theta, self.miu_zero, self.miu_one, self.sigma_zero, self.sigma_one = theta, miu_zero, miu_one, sigma_zero, sigma_one

		# 计算属性掌握概率
		self.know_prob = np.dot(post, self.all_states)

