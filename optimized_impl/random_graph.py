import random

# graph is undirected
# graph is represented as a list of m records
# each record contains two nodes and their interaction energy (i, j, vij)

def random_graph(n, p):
	graph = []			
	interaction_energy = 1 # initially assume that the energy is the same for each edge

	for i in range(n):
		for j in range(i+1,n):
			r = random.random()
			if r<=p:
				edge_i=[i,j,interaction_energy]
				edge_j=[j,i,interaction_energy]	# if j is a neighbor of i, then i is a neighbor of j
				graph.append(edge_i)
				graph.append(edge_j)
	return graph