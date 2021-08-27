# 使用EM算法计算a, b, c, d参数
# 但参数计算过程没法化简，最后参数更新规则很复杂
# 根据参数大致区间，遍历该区间的值，找到使导数最接近0的值

# DDina: Dynamic guess and slip dina
# dina guess and slip change with time
# guess = a*arctan(b*t)
# slip = c*arccot(d*t)


import pickle, math, time
import numpy as np
import pandas as pd
from sympy import acot

from dina import Dina


# 初始化模型参数
def init_parameters(stu_num, prob_num):
	theta = np.zeros(shape=stu_num)

	#guess_a = np.zeros(shape=prob_num)
	#guess_b = np.zeros(shape=prob_num)
	#slip_a = np.zeros(shape=prob_num)
	#slip_b = np.zeros(shape=prob_num)
	
	params = np.loadtxt('../data/abcd.csv', delimiter=',')
	guess_a = params[0]
	guess_b = params[1]
	slip_a = params[2]
	slip_b = params[3]
	#print(guess_a, guess_b, slip_a, slip_b)

	return theta, guess_a, guess_b, slip_a, slip_b

def guess_func(t, a, b):
	return a*np.arctan(b*t)

def arccot(x):
	return np.pi/2 - np.arctan(x)

def slip_func(t, a, b):
	#return a*(np.pi/2 - np.arctan(b*t))
	#return a * acot(b*t)
	return a * arccot(b*t)


class D1_Dina(Dina):
	def __init__(self, Q, R, T):
		super().__init__(Q, R)
		self.T = T
		print('max time:', np.max(T, axis=0))
		print('min time:', np.min(T, axis=0))
		self.theta, self.guess_a, self.guess_b, self.slip_a, self.slip_b = init_parameters(self.stu_num, self.prob_num)


	# EM算法估计模型参数
	def train(self, epoch, epsilon, duration=1, max_time=20): # duration: 每个时间段长度
		like = np.zeros(shape=(self.stu_num, self.state_num))  # 学生在每种属性的似然 
		post = np.zeros(shape=(self.stu_num, self.state_num))  # 学生在每种属性的后验概率 

		theta, guess_a, guess_b, slip_a, slip_b, tmp_R = np.copy(self.theta), \
				np.copy(self.guess_a), np.copy(self.guess_b), np.copy(self.slip_a), \
				np.copy(self.slip_b), np.copy(self.R)

		for iteration in range(epoch):
			post_tmp, guess_a_tmp, guess_b_tmp, slip_a_tmp, slip_b_tmp = np.copy(post), \
					np.copy(guess_a), np.copy(guess_b), \
					np.copy(slip_a), np.copy(slip_b)


			alphas = np.loadtxt('../data/more_alpha.csv', delimiter=',')
			alphas = alphas[:6000000]
			for i in range(post.shape[0]):
				alpha = alphas[i]
				j = int(alpha[0] * 4 + alpha[1] * 2 + alpha[2] * 1)
				post[i][j] = 1

			# 计算导数值
			# a
			t1 = guess_a * np.arctan(guess_b * self.T)
			t2 = (self.R - t1) / (guess_a * (1 - t1) + 1e-9)
			t3 = np.dot(np.transpose(post), t2)
			t = np.sum(t3 * (1 - self.eta), axis=0) 
			print(t)

			# b
			t1 = (guess_a * self.T) / (1 + (guess_b * self.T)**2)
			t2 = guess_a * np.arctan(guess_b * self.T)
			t3 = t1 * (self.R - t2) / (t2 * (1 - t2) + 1e-9)
			t4 = np.dot(np.transpose(post), t3)
			t = np.sum(t4 * (1 - self.eta), axis=0) 
			print(t)

			# c
			t1 = slip_a * arccot(slip_b * self.T)
			t2 = (self.R - 1 + t1) / (slip_a * (1 - t1) + 1e-9)
			t3 = np.dot(np.transpose(post), t2)
			t = np.sum(-t3 * self.eta, axis=0) 
			print(t)

			# d
			t1 = (slip_a * self.T) / (1 + (slip_b * self.T)**2)
			t2 = slip_a * arccot(slip_b * self.T)
			t3 = t1 * (self.R - 1 + t2) / (t2 * (1 - t2) + 1e-9)
			t4 = np.dot(np.transpose(post), t3)
			t = np.sum(t4 * self.eta, axis=0) 
			print(t)
			exit()

			# 获取最可能的属性组合
			theta = np.argmax(post, axis=1)

			change = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(guess_a - guess_a_tmp)), \
					np.max(np.abs(guess_b - guess_b_tmp)), np.max(np.abs(slip_a - slip_a_tmp)), \
					np.max(np.abs(slip_b - slip_b_tmp)))

			if iteration > 20 and change < epsilon:
				break

		self.theta, self.slip, self.guess = theta, slip, guess
		self.guess_a, self.guess_b, self.slip_a, self.slip_b = guess_a, guess_b, slip_a, slip_b

		# 计算属性掌握概率
		self.know_prob = np.dot(post, self.all_states)


if __name__ == '__main__':
	q = np.loadtxt('../data/q.csv', dtype=int, delimiter=',') 		# 读取Q矩阵
	d = np.loadtxt('../data/more_correct_wrong.csv', dtype=int, delimiter=',')	# 读取作答对错
	rt = np.loadtxt('../data/more_time.csv', delimiter=',')	# 读取作答时间

	d1_dina = D1_Dina(q, d, rt)
	d1_dina.train(1, 0.001)
	#print(d1_dina.guess_a, d1_dina.guess_b)
	#print(d1_dina.slip_a, d1_dina.slip_b)
	#for i, theta in enumerate(d1_dina.theta[:10]):
	#	print(d1_dina.all_states[theta])

