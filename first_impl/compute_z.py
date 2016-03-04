import numpy as np 
import random
import time
from subgraphs_world_generator import *

def compute_z(graph, n):
	# INIT #
	B = 20								# external field
	beta = 0.1  						# level of rationality
	epsilon = 0.95						# desired accuracy
	eta = float(1)/(4*n)	
	csi = epsilon/(2*n)
	s=int(72*np.power(csi,-2)*10)		# size of each set of samples from the generator
	#print "s ",s
	t=int(6*np.ceil(np.log(1/eta))+1)  	# number of repetitions of each generator experiment
	#delta = csi*(1.0/10)/8				# tolerance
	delta = epsilon/(160*n)
	m = len(graph)					# number of edges
	lamb = [] 						# lamb[m] = lambda_ij for each edge m
	mu = []							# mu contains values mu_k 
	
	for i in range(m):
		edge = graph[i]
		lamb.append(np.tanh(beta*edge[2]))


	# FIRST STEP #
	A = np.power((2*np.cosh(beta*B)),n)		# compute A
	Z_first = 1.0							# compute Z'(1)
	prod = 1.0
	for i in range(m):
		edge = graph[i]
		prod *= np.cosh(beta*edge[2])
		Z_first *= (1+lamb[i])
	A *= prod


	# SECOND STEP #
	k = 0
	final_mu = np.tanh(beta*B)
	k_mu = (n-k)/float(n)
	while k_mu > final_mu:
		mu.append(k_mu)
		k += 1
		k_mu = (n-k)/float(n)	
	mu.append(final_mu)				# fill array of mu values until mu=tanh(beta*B) is reached
	size_mu = len(mu)
	Y = 1.0
	for k in range(size_mu-1):
		median = compute_subgraphs_configurations(graph, n, s, t, mu[k], mu[k+1], lamb, delta)
		Y *= median


	# THIRD STEP #
	Z = A * Z_first * Y
	return Z 



def compute_subgraphs_configurations(graph, n, s, t, mu, next_mu, lamb, delta):
	means = []
	func = float(next_mu)/mu
	for tm in range(t):
		values = 0.0
		sample_mean = 0
		for element in xrange(s):
			#start = time.clock()
			parity = subgraphs_world_generator(graph, n, mu, delta, lamb)
			#end = time.clock()
			#print end - start
			odd_X = 0
			for i in range(len(parity)):
				odd_X += parity[i]
			f_X = np.power(func,odd_X)
			values += f_X

		sample_mean = values/float(s)
		means.append(sample_mean)

	median = np.median(means)
	return median