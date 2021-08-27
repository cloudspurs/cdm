import numpy as np
from itertools import product


class Cdm():
	"""
		Q: Q矩阵
		R: 作答对错矩阵
		stu_num: 学生数量
		prob_num: 题目数量
		know_num: 属性数量
	"""
	def __init__(self, Q, R):
		self.Q, self.R = Q, R
		self.stu_num, self.prob_num, self.know_num = R.shape[0], R.shape[1], Q.shape[1] 
		self.state_num = 2**self.know_num

		self.all_states = np.array(list(product([0, 1], repeat=self.know_num)))

		state_prob = np.transpose(np.sum(Q, axis=1, keepdims=True) - np.dot(Q, np.transpose(self.all_states)))
		self.eta = 1 - (state_prob > 0) 


if __name__ == '__main__':
	q = np.loadtxt('../data/q.csv', dtype=int, delimiter=',') 		# 读取Q矩阵
	r = np.loadtxt('../data/correct_wrong_dina.csv', dtype=int, delimiter=',')	# 读取作答对错
	cdm = Cdm(q, r)
	print('all states\n', cdm.all_states)
	print('eta\n', cdm.eta)

