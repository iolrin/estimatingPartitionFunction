import numpy as np 
import time
import random
from subgraphs_world_generator import *
from joblib import Parallel, delayed

def compute_z(graph, n, B, beta, epsilon, num_jobs):
	# INIT #
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

	r = len(mu)-1 						# we use r instead of n for calculating eta and csi...
	eta = float(1)/(4*r)				# ...we can do this because r is bounded by n
	csi = epsilon/(2*r)
	t = 2						# number of repetitions of each generator experiment
	
	Y = 1.0
	for k in range(r):
		min_f = np.power((mu[k+1]/mu[k]), n)
		E_f = np.power((mu[k+1]/mu[k]), 2*np.floor(n/2))
		s = 0
		Gamma_csi_f = 0
		delta = 0
		if 1-min_f <= csi/2:
			c = (csi - 1 + min_f)/(E_f +1 -min_f)
			Gamma_csi_f_num = (c+1)*(1-min_f) + (csi-c)*(3+min_f)
			Gamma_csi_f_den = np.power((2*c-csi+1),2)/(c+1)
			Gamma_csi_f = Gamma_csi_f_num/Gamma_csi_f_den
			s = int(np.ceil(4*np.power(c,-2)*(Gamma_csi_f/E_f)))
			delta = ((csi-c)*E_f)/((c+1)*(1-min_f)) 
		else:
			c = csi /(2+csi)
			Gamma_csi_f = ((1-min_f) + (csi/2)*(3+min_f)) / np.power(1-csi/2,2)
			s = int(np.ceil(4*np.power(c,-2)*(Gamma_csi_f/E_f)))
			delta = (csi*E_f)/(2*(1-min_f))
		# print "csi ",csi
		# print "c ",c
		# print "E_f ",E_f
		# print "min_f ",min_f
		# print "Gamma_csi_f ",Gamma_csi_f
		# print "s ",s
		# print "delta ",delta
		median = compute_subgraphs_configurations(graph, n, s, t, mu[k], mu[k+1], lamb, delta, num_jobs)
		Y *= median


	# THIRD STEP #
	Z = A * Z_first * Y
	return Z,s,t 


def compute_subgraphs_configurations(graph, n, s, t, mu, next_mu, lamb, delta, num_jobs):
	means = []
	func = float(next_mu)/mu
	for tm in range(t):
		functions = Parallel(n_jobs=num_jobs)(delayed(compute_generator)(func, graph, n, mu, delta, lamb) for step in range(s))
		means.append(np.mean(functions))

	median = np.median(means)
	return median


def compute_generator(func, graph, n, mu, delta, lamb):
	parity = subgraphs_world_generator(graph, n, mu, delta, lamb)
	odd_X = 0
	for i in range(len(parity)):
		odd_X += parity[i]
	f_X = np.power(func,odd_X)
	return f_X