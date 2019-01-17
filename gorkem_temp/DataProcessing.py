
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readData():
	# Read the data
	relations = pd.read_csv('filtered_relations.csv')
	users = pd.read_csv('filtered_users.csv')
	return relations, users

def displayInfo(datainput, inputname):
	display(datainput.head())
	#display(datainput.shape[0])
	print("We have ",str(datainput.shape[0])," ", inputname,"s.")
	print("Stats info about ",inputname,":")
	display(datainput.describe())

def printFurtherInfo(relations):
	# Is there unknown/NA values?  
	print("Is there any NA? \n____________")
	display(relations.isnull().sum())
	# Relation type 
	print("\nMake sure we only have relation type 5 as data \n____________")
	display(relations.relation.unique())
	# Unique days
	print("\nWhat are the unique day values in relations data? \n____________")
	display(relations.day.unique())

def createNodes(users):
	#Define Nodes
	nodes = users[['User Id', "Time", "Gender", "Age Range", "Spammer Label"]]
	num_nodes = len(nodes)
	nodes.reset_index(level=0, inplace=True)
	nodes = nodes.rename(columns={'index':'node_idx'})
	nodes.head(3)
	return nodes

def createEdges(nodes, relations):
	#Define Edges
	edges = relations[["src", "dst", "relation"]]
	num_edges = len(edges)
	edges.head(3)

	# Create a conversion table from User Id to node index.
	uid2idx = nodes[['node_idx', 'User Id']]
	uid2idx = uid2idx.set_index('User Id')
	uid2idx.index.name = 'src'
	uid2idx.head()

	# Add a new column, matching the "src" column with the node_idx.
	# Do the same with the "dst" column.
	edges = edges.join(uid2idx, on="src")
	edges = edges.join(uid2idx, on='dst', rsuffix='_dst')
	edges.head()
	# Drop the src, dst.
	edges = edges.drop(columns=['src','dst'])
	return edges

def createAdjacencyMatrix(nodes, edges, isDirected):
	num_nodes = len(nodes)
	if isDirected:
		# We build the adjacency matrix with int8 in order to save on memory resources. 
		adjacency = np.zeros((num_nodes, num_nodes), dtype = np.dtype('>i1'))
		for idx, row in edges.iterrows():
		    i, j = int(row.node_idx), int(row.node_idx_dst)
		    adjacency[i, j] = 1
		n_nodes = num_nodes
		return adjacency
	else:
		# Creating undirected graph in a different numpy array
		adjacency = np.zeros((num_nodes, num_nodes), dtype = np.dtype('>i1'))
		for idx, row in edges.iterrows():
		    i, j = int(row.node_idx), int(row.node_idx_dst)
		    adjacency[i, j] = 1
		n_nodes = num_nodes
		undirected_adjacency = adjacency.copy()
		for idx, row in edges.iterrows():
		    i, j = int(row.node_idx), int(row.node_idx_dst)
		    undirected_adjacency[j, i] = 1
		return undirected_adjacency

def calculateIn_Out_Degrees(edges, num_nodes):
	in_degree = np.zeros((num_nodes, 1), dtype=int)
	out_degree = np.zeros((num_nodes, 1), dtype=int)
	for idx, row in edges.iterrows():
	    src, dst = int(row.node_idx), int(row.node_idx_dst)
	    out_degree[src] += 1
	    in_degree[dst] += 1
	degree =  in_degree + out_degree 
	return in_degree, out_degree, degree

def calculateAvgDegree(in_degree, out_degree, num_nodes):
	Lin = np.sum(in_degree)
	Lout = np.sum(out_degree)
	assert Lin == Lout
	print('L is: ',Lin)
	# Since Lin is equal to Lout we can use either one as L in the below calculation
	avg_degree = Lin /  num_nodes
	print('average degree is: ',avg_degree)
	return avg_degree

def isConnected(adjacency):
	touched_nodes = np.zeros(adjacency.shape[0])
	touched_nodes = start_dfs(adjacency, touched_nodes, 0)
	return (adjacency.shape[0] == np.sum(touched_nodes))

def visit(adjacency, visited_nodes, current_node):
    neighbors = adjacency[current_node]      
    for i in np.nonzero(neighbors)[0]:       
        if (visited_nodes[i] == 0): 
            visited_nodes[i] = 1;
            visited_nodes = visit(adjacency, visited_nodes, i) 
    return visited_nodes

def start_dfs (adjacency, visited_nodes, current_node ):
    visited_nodes_1 = np.copy(visited_nodes)
    visited_nodes_1[current_node] = 1 
    return visit(adjacency, visited_nodes_1,current_node)

def compute_clustering_coefficient(adjacency, node):
    neigh =  np.nonzero(adjacency[node])[0]
    if len(neigh) == 0 or len(neigh) == 1:
        return 0
    links = 0
    count =0
    for v in neigh:
        count+=1
        neighV =  np.nonzero(adjacency[v])[0]
        for u in neigh:
            if u in neighV: 
                # We increment by 0.5 in order to take into account double-counting
                links += 0.5
    clustering_coefficient = 2.0 * links / (len(neigh)*(len(neigh)-1))
    return clustering_coefficient
