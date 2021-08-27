# 遍历0-1求得guess和slip，结果显示可行，和EM估计的参数差不多
import pickle
import numpy as np
from cdm import Cdm


# 初始化模型参数
def init_parameters(stu_num, prob_num):
	slip = np.zeros(shape=prob_num) + 0.2
	guess = np.zeros(shape=prob_num) + 0.2
	theta = np.zeros(shape=stu_num)
	return theta, slip, guess


class Dina(Cdm):
	def __init__(self, Q, R):
		super().__init__(Q, R)
		self.theta, self.slip, self.guess = init_parameters(self.stu_num, self.prob_num)
		

	# EM算法估计模型参数
	def train(self, epoch, epsilon):
		like = np.zeros(shape=(self.stu_num, self.state_num))  # 学生在每种属性的似然 
		post = np.zeros(shape=(self.stu_num, self.state_num))  # 学生在每种属性的后验概率 

		theta, slip, guess, tmp_R = np.copy(self.theta), np.copy(self.slip), np.copy(self.guess), np.copy(self.R)

		#print(self.all_states)
		#print(self.Q)
		#print(self.eta)
		for iteration in range(epoch):
			post_tmp, slip_tmp, guess_tmp = np.copy(post), np.copy(slip), np.copy(guess)

			# 题目答对概率
			answer_right = (1 - slip) * self.eta + guess * (1 - self.eta)
			#print(answer_right)

			# 每个状态下，每个学生在所有题目的似然
			for s in range(self.state_num):
				# 对数似然
				#ttt = answer_right[s,:] * self.R + (1-answer_right[s,:]) * (1-self.R)
				#print(self.R[:2])
				#print(ttt[:2])
				log_like = np.log(answer_right[s, :] + 1e-9) * self.R + np.log(1 - answer_right[s, :] + 1e-9) * (1 - self.R)

				# 每个状态下，每个学生所有题目的似然和
				like[:, s] = np.exp(np.sum(log_like, axis=1))

			post = like / np.sum(like, axis=1, keepdims=True) # 学生在每种状态的后验概率

			## 每种状态下的学生人数
			#i_l = np.expand_dims(np.sum(post, axis=0), axis=1)
			## 每种状态下，eta=0 和 eta=1的人数
			#i_jl_0, i_jl_1 = np.sum(i_l * (1 - self.eta), axis=0), np.sum(i_l * self.eta, axis=0)
			## 每种状态下每个题目的答对人数
			#r_jl = np.dot(np.transpose(post), tmp_R)
			## 每个题目，eta=0答对的人数和eta=1答对的人数
			#r_jl_0, r_jl_1 = np.sum(r_jl * (1 - self.eta), axis=0), np.sum(r_jl * self.eta, axis=0)
			##print(i_jl_0, i_jl_1, r_jl_0, r_jl_1)
			## 更新guess, slip参数
			#guess, slip = r_jl_0 / i_jl_0, (i_jl_1 - r_jl_1) / i_jl_1

			# 遍历求参数
			candidates = np.linspace(0, 1, 100)

			# guess
			derivatives = []
			for g in candidates:
				t1 = (self.R - g) / (g * (1 - g) + 1e-9)
				t2 = np.dot(np.transpose(post), t1)
				t = np.sum(t2 * (1 - self.eta), axis=0)
				derivatives.append(t)
			derivatives = np.array(derivatives)
			index = np.argsort(np.abs(derivatives-0), axis=0)
			guess = candidates[index[0]]

			# slip
			derivatives = []
			for s in candidates:
				t1 = (self.R - (1 - s)) / (slip * (1 - slip) + 1e-9)
				t2 = np.dot(np.transpose(post), t1)
				t = np.sum(t2 * self.eta, axis=0)
				derivatives.append(t)
			derivatives = np.array(derivatives)
			index = np.argsort(np.abs(derivatives-0), axis=0)
			slip = candidates[index[0]]

			# 计算导数 guess
			#t1 = (self.R - guess) / (guess * (1 - guess))
			#t2 = np.dot(np.transpose(post), t1)
			#t = np.sum(t2 * (1 - self.eta), axis=0)
			#print(t)

			## 计算导数 slip
			#t1 = (self.R - (1 - slip)) / (slip * (1 - slip))
			#t2 = np.dot(np.transpose(post), t1)
			#t = np.sum(t2 * self.eta, axis=0)
			#print(-t)

			# 获取最可能的属性组合
			theta = np.argmax(post, axis=1)

			change = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(slip - slip_tmp)), np.max(np.abs(guess - guess_tmp)))
			if iteration > 20 and change < epsilon:
				break

		self.theta, self.slip, self.guess = theta, slip, guess

		# 计算属性掌握概率
		self.know_prob = np.dot(post, self.all_states)


if __name__ == '__main__':
	q = np.loadtxt('../data/q.csv', dtype=int, delimiter=',') 		# 读取Q矩阵
	d = np.loadtxt('../data/correct_wrong_dina.csv', dtype=int, delimiter=',')	# 读取作答对错

	dina = Dina(q, d)
	dina.train(1000, 0.001)
	print('guess', dina.guess)
	print('slip', dina.slip)
	#print('theta')
	#for i, theta in enumerate(dina.theta[30:50]):
	#	print(dina.all_states[theta])
