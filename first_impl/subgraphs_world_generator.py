import numpy as np 
import random

# graph: the graph of the social network, represented as a list of m records (i,j,Vij)
# n: the number of nodes in the graph
# beta: the rationality level
# B: the external field
# delta: the tolerance
def subgraphs_world_generator(graph, n, mu, delta, lamb):
	m = len(graph)		# number of total edges in the graph
	X = []				# X[m]=1 if the edge graph[m] is in the subgraph; X[m]=0 if the edge graph[m] is not in the subgraph
	parity = []			# parity[i]=1 if node i has odd degree in X; parity[i]=0 if node i has even degree in X

	#mu = np.tanh(beta*B)
	# the total numer of steps to simulate the Markov chain is 16*m^2*mu^-8*(ln(delta^-1)+m)
	number_of_steps = int(16*np.power(m,2)*np.power(mu,-8)*(np.log(1/delta)+m))# the number of step for simulating Markov Chain

	for i in range(m):
		X.append(0)			# the chain starts with the empty subgraph, so X[m]=0 for each edge m...

	for i in range(n):
		parity.append(0)	# ...and the parity is 0 (even) for each node


	for step in xrange(number_of_steps):
		r = random.random()			
		if r > 0.5: 
			k = random.randint(0,m-1)
			edge = graph[k]	# select an edge uniformly at random 
			i = edge[0]
			j = edge[1]
			vij = edge[2] 	# interaction energy
			exp = 0
			if parity[i] == 0 and parity[j] == 0: # if both nodes have even degree in X, now with this edge they have odd degree
				exp = 2							  # ...so we add them
			if parity[i] == 1 and parity[j] == 1: # if both nodes have odd degree in X, now with this edge they have even degree
				exp = -2						  # ...so we remove them
			lambda_ij = lamb[k]

			# difference is the ratio W(Y)/W(X)
			if X[k] == 0:
				difference = lambda_ij * (np.power(mu,exp))
			else:
				difference = (1/lambda_ij) * (np.power(mu,exp))
				
			# difference >= 1 means that W(Y) >= W(X)
			if difference >= 1:		# edge k is added or removed, so i have to change the parity of the nodes
				X[k] = (X[k]+1) % 2
				parity[i] = (parity[i]+1) % 2
				parity[j] = (parity[j]+1) % 2
			else: # means that W(Y) < W(X), this happens when difference is < 1
				r = random.random()		# edge k is added with probability difference=W(Y)/W(x)
				if r <= difference:
					X[k] = (X[k]+1) % 2
					parity[i] = (parity[i]+1) % 2
					parity[j] = (parity[j]+1) % 2

	return parity
