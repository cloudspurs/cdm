1. files
	src/
		cdm.py: 所有模型的基类，初始化所有的属性向量和eta
		dina.py: Dina模型
		dina_tranverse.py: 遍历guess和slip的值，求最小导数的guess，slip
		c_dina.py: C-Dina模型
		mt_dina.py: 融合时间的Dina模型, 对数据按时间进行划分，求每段数据guess和slip，再根据每段时间的平均值拟合guess，slip曲线参数
		mt_dina_1.py: 先划分数据，求每段数据的guess和slip，再根据每段时间的平均值拟合guess，slip曲线参数
		mt_dina_2.py: 每次迭代，既计算每段数据的guess和slip，又计算guess，slip曲线参数
		one_question_dina.py: 每次只取一个题目一个时间段的数据计算gues和slip
		t_dina.py: 先用C-Dina估计题目参数，再在Dina模型的似然然后中加入C-Dina模型的概率密度函数
		x_dina.py: 似然函数为Dina和C-Dina的似然和，一起估计Dina和C-Dian题目参数，效果一般
		d1_dina.py: 遍历求解参数值
		d1_dina_daoshu.py：带入真实值，验证导数大小
		d2_dina.py: 遍历求解参数值，guess，slip与时间只有一个参数
		jags_jrt_dina.py: 使用JAGS包估计JRT-Dina模型参数，需先用apt install安装JAGS包，再用python接口调用
		jrt_dina.jags: JRT-Dina模型定义文件
		simulate_data_c_dina.py: 先用Dina估计属性向量，再模拟生成C-Dina数据
		simulate_data_mt_dina: 模拟生成MT-Dina数据，属性向量，时间等随机生成，最后生成对错
		statics.py: 统计生成数据的真实guess和slip
	data/
		q.csv: Q矩阵（来源于R G-Dina包模拟数据）
		alpha.csv: 模拟生成的学生属性向量
		correct_wrong.csv: 根据MT-Dina模型模拟生成的作答对错
		correct_wrong_dina.csv: Dina作答对错数据（来源于R G-Dina包模拟数据）
		time.csv: 模拟生成的MT-Dina作答时间
		time_c_dina.csv: 模拟生成的C-Dina作答时间
		abcd.csv: 模拟生成的a, b, c, d参数

2. 基本算法
	2.1 缺失数据处理
		可以直接把缺失数据看作0或1。只适用于少量缺失数据的情况下，大量缺失
		数据严重影响参数估计结果。
		当一个人没有作答数据时，应直接忽略这个人，所有的后验概率应都改为零？
		
	2.2 分段数据处理
		E步可以放在一起计算，根据不同的数据段使用不同的参数。
		M步更新参数时，更新每段数据的参数，只是用该段数据对应的后验概率。

