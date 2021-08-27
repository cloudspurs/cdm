# 使用EM算法计算a, b, c, d参数
# 但参数计算过程没法化简，最后参数更新规则很复杂
# 根据参数大致区间，遍历该区间的值，找到使导数最接近0的值

# DDina: Dynamic guess and slip dina
# dina guess and slip change with time

# guess和slip函数修改为只有一个参数
# guess = a*arctan(t)
# slip = c*arccot(t)


import pickle, math, time
import numpy as np
import pandas as pd
from sympy import acot

from dina import Dina


# 初始化模型参数
def init_parameters(stu_num, prob_num):
	theta = np.zeros(shape=stu_num)
	guess_a = np.zeros(shape=prob_num) + 0.1
	guess_b = np.zeros(shape=prob_num) 
	slip_a = np.zeros(shape=prob_num) + 0.1
	slip_b = np.zeros(shape=prob_num)
	return theta, guess_a, guess_b, slip_a, slip_b

def guess_func(t, a):
	return a*np.arctan(t)

def arccot(x):
	return np.pi/2 - np.arctan(x)

def slip_func(t, c):
	#return a*(np.pi/2 - np.arctan(b*t))
	#return a * acot(b*t)
	return c * arccot(t)


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

			# 根据slope和intercept计算guess和slip, 每个人每个题上的guess和slip
			guess = guess_func(self.T, guess_a)
			slip = slip_func(self.T, slip_a)

			#print(guess[:10])
			#print(slip[:10])
			
			guess = np.where(guess > 1, 1, guess)
			guess = np.where(guess < 0, 0, guess)
			slip = np.where(slip > 1, 1, slip)
			slip = np.where(slip < 0, 0, slip)

			# 1. 计算后验概率（每个状态下，每个学生在所有题目的似然）
			for s in range(self.state_num):
				answer_right = guess * (1 - self.eta[s]) + (1 - slip) * self.eta[s]
				log_like = np.log(answer_right + 1e-9) * self.R + np.log(1 - answer_right + 1e-9) * (1 - self.R)
				like[:, s] = np.exp(np.sum(log_like, axis=1))
			post = like / np.sum(like, axis=1, keepdims=True) # 学生在每种状态的后验概率
			#print(post)

			# 开始更新参数
			start = time.time()

			# 遍历a,获取导数最接近0的a
			candidates= np.linspace(0.1, 2, 100)

			# guess_a
			derivatives = []		
			for a in candidates:
				t1 = a * np.arctan(self.T)
				t2 = (self.R - t1) / (a * (1 - t1) + 1e-9)
				t3 = np.dot(np.transpose(post), t2)
				t = np.sum(t3 * (1 - self.eta), axis=0) 
				derivatives.append(t)
			derivatives = np.array(derivatives)
			index = np.argsort(derivatives-0, axis=0)
			guess_a = candidates[index[0]]

			#print(candidates[index[0]])
			#xxx = [derivatives[index[0, i], i] for i in range(self.prob_num)]
			#print(xxx)

			# slip_a
			derivatives = []		
			for c in candidates:
				t1 = c * arccot(self.T)
				t2 = (self.R - 1 + t1) / (c * (1 - t1) + 1e-9)
				t3 = np.dot(np.transpose(post), t2)
				t = np.sum(-t3 * self.eta, axis=0) 
				derivatives.append(t)
			derivatives = np.array(derivatives)
			index = np.argsort(np.abs(derivatives), axis=0)
			slip_a = candidates[index[0]]

			#print(candidates[index[0]])
			#xxx = [derivatives[index[0, i], i] for i in range(self.prob_num)]
			#print(xxx)

			end = time.time()
			#print('time', end-start)
			#print(guess_a)
			#print(guess_b)

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
	d = np.loadtxt('../data/correct_wrong.csv', dtype=int, delimiter=',')	# 读取作答对错
	rt = np.loadtxt('../data/time.csv', delimiter=',')	# 读取作答时间

	# 小数据
	#q = np.loadtxt('../data/small_q.csv', dtype=int, delimiter=',') 		# 读取Q矩阵
	#d = np.loadtxt('../data/small_correct_wrong.csv', dtype=int, delimiter=',')	# 读取作答对错
	#rt = np.loadtxt('../data/small_time.csv', delimiter=',')	# 读取作答时间

	d1_dina = D1_Dina(q, d, rt)
	d1_dina.train(100, 0.001)
	print(d1_dina.guess_a, d1_dina.guess_b)
	print(d1_dina.slip_a, d1_dina.slip_b)
	#for i, theta in enumerate(d1_dina.theta[:10]):
	#	print(d1_dina.all_states[theta])

