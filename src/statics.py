import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
from d1_dina import arccot

# 每道题，eta=0，eta=1两种情况，以1s为区间，做对题目个数和比例

Q = np.loadtxt('../data/q.csv', dtype=int, delimiter=',')	# 读取Q矩阵
alphas = np.loadtxt('../data/alpha.csv', delimiter=',')
r = np.loadtxt('../data/correct_wrong.csv', dtype=int, delimiter=',')	# 读取作答对错
t = np.loadtxt('../data/time.csv', delimiter=',')	# 读取作答时间

prob = np.transpose(np.sum(Q, axis=1, keepdims=True) - np.dot(Q, np.transpose(alphas)))
eta = 1 - (prob > 0)

params = np.loadtxt('../data/abcd.csv', delimiter=',')
guess_a = params[0]
guess_b = params[1]
slip_a = params[2]
slip_b = params[3]

for i in range(Q.shape[0]):
	#a = []
	#for k, g in groupby(sorted(t[:,i]), key=lambda x: x//1):
	#	a.append(len(list(g)))

	all_eta_zero = []
	all_eta_one = []
	all_zero_right = []
	all_one_right = []
	for j in range(20):
		eta_zero = 0
		eta_one = 0
		eta_zero_right = 0
		eta_one_right = 0
		for k, time in enumerate(t[:, i]):
			if j < time <= j+1:
				if eta[k, i]  == 0:
					eta_zero += 1
					if r[k, i] == 1:
						eta_zero_right += 1
				if eta[k, i]  == 1:
					eta_one += 1
					if r[k, i] == 1:
						eta_one_right += 1
		all_eta_zero.append(eta_zero)
		all_eta_one.append(eta_one)
		all_zero_right.append(eta_zero_right)
		all_one_right.append(eta_one_right)
	print(all_eta_zero)
	print(all_eta_one)
	print(all_zero_right)
	print(all_one_right)
	x = range(len(all_eta_zero))

	#plt.bar(x, all_eta_zero)
	#plt.bar(x, all_zero_right)

	#for a, b in enumerate(np.array(all_zero_right)/np.array(all_eta_zero)):
	#	plt.text(a-0.5, all_eta_zero[a]+1, "%.2f"%b, fontsize=8)
	#plt.savefig('question_' + str(i) + '_eta0.png')
	#plt.clf()

	y = np.array(all_zero_right)/np.array(all_eta_zero)
	plt.plot(x, y, label='guess generate')

	x = np.linspace(0, 20, 100)
	y = guess_a[i] * np.arctan(guess_b[i] * x)
	plt.plot(x, y, label='guess real')

	#plt.bar(x, all_eta_one)
	#plt.bar(x, all_one_right)
	#for a, b in enumerate(np.array(all_one_right)/np.array(all_eta_one)):
	#	plt.text(a-0.5, all_eta_one[a]+1, "%.2f"%b, fontsize=8)
	#plt.savefig('question_' + str(i) + '_eta1.png')
	#plt.clf()

	x = range(len(all_eta_zero))
	y = (np.array(all_eta_one) - np.array(all_one_right)) / np.array(all_eta_one)
	plt.plot(x, y, label='slip generate')

	x = np.linspace(0, 20, 100)
	y = slip_a[i] * arccot(slip_b[i] * x)
	plt.plot(x, y, label='slip real')

	plt.legend(loc='best')

	plt.savefig('../data/img/real_' + str(i) + '.png')
	plt.clf()


				

