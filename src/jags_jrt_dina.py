import os 
import pyjags
import numpy as np
from itertools import product
import savReaderWriter

#print(np.exp(-3.6)/(1+np.exp(-3.6)))
#exit()

# paper dataset
#reader = savReaderWriter.SavReader('xx.sav')
#print(reader)
#header = reader.next()
#print(header, '\n')
#
#stus = set()
#for line in reader:
#	c = line[0].decode()
#	stu_id = line[3].decode()
#	if c in ['USA']:
#		stus.add(stu_id)
#print(len(stus))
#exit()


#know_num = 3
#all_states = np.array(list(product([0, 1], repeat=know_num)))

Q = np.loadtxt('q.txt', delimiter=',') 
x = np.loadtxt('dat.txt', delimiter=',')
response_time = np.loadtxt('simulate.csv', delimiter=',')
N = x.shape[0]
I = x.shape[1]
K = Q.shape[1]

pyjags.load_module('glm')

path = os.path.dirname(__file__)
path = os.path.join(path, 'jrt_dina.jags')

# model data
data = dict(N=N, I=I, K=K, Q=Q, score=x, log_rt=np.log(response_time))

model = pyjags.Model(file=path, data=data, chains=1, adapt=1000)

model.sample(5000, vars=[])

# model monitor variables
samples = model.sample(5000, vars=['attribute', 'beta', 'delta', 'zeta', 'tao', 'gamma', 'theta', 'lanmda', 'guess', 'slip', 'ppp'])
#samples = model.sample(100, vars=['attribute', 'theta', 'guess', 'slip'])

#print(samples['attribute'].shape) 
print(np.squeeze(samples['attribute'][:20,:,-1,:])) # shape: (1000, 3, 5000, 1)

print(samples['guess'][:,-1,:]) # shape: (10, 5000, 1)
#print(np.mean(np.squeeze(samples['guess']), axis=1)) # shape: (10, 5000, 1)
guesss = np.mean(np.squeeze(samples['guess']), axis=1)
print('guess')
for g in guesss:
	print("%.3f" % g)

print(samples['slip'][:,-1,:]) # shape: (10, 5000, 1)
#print(np.mean(np.squeeze(samples['slip']), axis=1)) # shape: (10, 5000, 1)
slips = np.mean(np.squeeze(samples['slip']), axis=1)
print('slip')
for s in slips:
	print("%.3f" % s)

print('PPP:', samples['ppp'][-1,-1])


# Initialize model with 4 chains and run 1000 adaptation steps in each chain.
# We treat alpha, beta and sigma as parameters we would like to infer, based
# on observed values of x and y.
#model = pyjags.Model(code, data=dict(x=x, y=y, N=N), chains=4, adapt=1000)
 
# 500 warmup / burn-in iterations, not used for inference.
#model.sample(500, vars=[])

# Run model for 1000 steps, monitoring alpha, beta and sigma variables.
# Returns a dictionary with numpy array for each monitored variable.
# Shapes of returned arrays are (... shape of variable ..., iterations, chains).
# In our example it would be simply (1, 1000, 4).
#samples = model.sample(1000, vars=['alpha', 'beta', 'sigma'])


