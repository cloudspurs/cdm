import time
import numpy as np
from sympy import acot


if __name__ == '__main__':
	Q = np.loadtxt('../data/q.csv', delimiter=',')

	batch_size = 100000
	stu_num = batch_size * 1
	mode = 'wb' # 生成新的数据
	#mode = 'ab' # 追加数据

	alpha_path = '../data/alpha.csv'
	time_path = '../data/time.csv'
	correct_wrong_path = '../data/correct_wrong.csv'

	prob_num = Q.shape[0]
	know_num = Q.shape[1]

	#随机生成a, b, c, d
	a, b, c, d = np.zeros((prob_num,)), np.zeros((prob_num,)), np.zeros((prob_num,)), np.zeros((prob_num,)) 
	a = np.random.uniform(0.1, 0.2, (10,))	
	b = np.random.uniform(0.2, 0.3, (10,))	
	c = np.random.uniform(0.1, 0.2, (10,))	
	d = np.random.uniform(0.2, 0.3, (10,))	
	np.savetxt('../data/abcd.csv', np.array([a, b, c, d]), delimiter=',', fmt='%.5f')

	max_num = int(stu_num / batch_size)

	for i in range(max_num):
		print(i+1, '/', max_num, ':', (i+1) * batch_size)

		# 随机生成属性向量
		alpha=np.random.randint(0, 2, (batch_size, know_num))
		etas = np.transpose(1 - ((np.sum(Q, axis=1, keepdims=True) - np.dot(Q, np.transpose(alpha))) > 0)) #潜在作答向量

		t = np.random.randint(1, 21, (batch_size, prob_num))
		correct_wrong = np.zeros((batch_size, prob_num))

		for i in range(batch_size):
			for j in range(prob_num):
				rand = np.random.uniform(0, 1)
				#生成作答对错
				if etas[i][j] == 0:
					if(a[j] * np.arctan(b[j] * t[i][j])) > rand:
						correct_wrong[i][j] = 1
				if etas[i][j] == 1:
					if(1 - (c[j] * acot(d[j] * t[i][j]))) > rand :
						correct_wrong[i][j] = 1

		with open(alpha_path, mode=mode) as f:
			np.savetxt(f, alpha, delimiter=',', fmt='%d')
		with open(time_path, mode=mode) as f:
			np.savetxt(f, t, delimiter=',', fmt='%d')
		with open(correct_wrong_path, mode=mode) as f:
			np.savetxt(f, correct_wrong,delimiter=',', fmt='%d')

