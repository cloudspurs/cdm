from cdm import Cdm

import numpy as np


class Dina(Cdm):
	def __init__(self, Q, R, g=0.2, s=0.2): # g, s: guess, slip初值
		super().__init__(Q, R)
		self.theta = np.zeros(shape=self.stu_num) # 属性向量
		self.guess = np.zeros(shape=self.prob_num) + g
		self.slip  = np.zeros(shape=self.prob_num) + s
		

	# EM算法估计模型参数
	def train(self, epoch, epsilon):
		like = np.zeros(shape=(self.stu_num, self.state_num))  # 似然：每个学生所有题目的似然
		post = np.zeros(shape=(self.stu_num, self.state_num))  # 后验概率：每个学生在每个属性向量的概率

		theta, slip, guess, R = self.theta.copy(), self.slip.copy(), self.guess.copy(), self.R.copy()

		for iteration in range(epoch):
			post_tmp, slip_tmp, guess_tmp = post.copy(), slip.copy(), guess.copy()

			answer_right = (1 - slip) * self.eta + guess * (1 - self.eta) # 每个属性向量下，每个题目答对的概率

			# 每个属性向量下，每个学生的所有题目的似然和
			for s in range(self.state_num):
				# 当前属性向量下，每个学生每个题目的对数似然
				log_like = np.log(answer_right[s, :] + 1e-9) * R + np.log(1 - answer_right[s, :] + 1e-9) * (1 - R)
				like[:, s] = np.exp(np.sum(log_like, axis=1)) # 当前属性向量下，每个学生所有题目的似然和
			post = like / np.sum(like, axis=1, keepdims=True) # 学生在每个属性向量的后验概率

			i_l = np.expand_dims(np.sum(post, axis=0), axis=1) # 每种属性向量下的学生人数
			i_jl_0, i_jl_1 = np.sum(i_l * (1 - self.eta), axis=0), np.sum(i_l * self.eta, axis=0) # 每个题eta=0和eta=1的人数

			r_jl = np.dot(np.transpose(post), R) # 每种属性向量下每个题目的答对人数
			# 每个题目，eta=0答对的人数和eta=1答对的人数
			r_jl_0, r_jl_1 = np.sum(r_jl * (1 - self.eta), axis=0), np.sum(r_jl * self.eta, axis=0)

			guess, slip = r_jl_0 / i_jl_0, (i_jl_1 - r_jl_1) / i_jl_1 # 更新guess, slip参数

			theta = np.argmax(post, axis=1) # 获取后验概率最大的属性向量

			change = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(slip - slip_tmp)), np.max(np.abs(guess - guess_tmp)))
			if iteration > 20 and change < epsilon:
				break

		self.theta, self.slip, self.guess = theta, slip, guess

		self.know_prob = np.dot(post, self.all_states) # 计算属性掌握概率


if __name__ == '__main__':
	q = np.loadtxt('../data/q.csv', dtype=int, delimiter=',')	# 读取Q矩阵
	d = np.loadtxt('../data/correct_wrong_dina.csv', dtype=int, delimiter=',')	# 读取作答对错

	dina = Dina(q, d)
	dina.train(1000, 0.001)

	print('guess', dina.guess)
	print('slip', dina.slip)
	print('theta')
	for i, theta in enumerate(dina.theta[:10]):
		print(dina.all_states[theta])

