import numpy as np
import networkx as nx
import random
random.seed(0)
class myGraph:
    def __init__(self, node_id):
        self.adjmat=0
        self.node=0
        self.edge=0
        self.id=node_id

    def setA(self):
        self.A=np.zeros((self.edge,self.node))
        k=0
        for i in range(self.node):
            for j in range(i, self.node):
                if (self.adjmat[i][j]):
                    self.A[k][i]=-1
                    self.A[k][j]=1
                    k=k+1
        d_0=np.diagflat(np.power(self.D,-0.5))
        t_DA=np.dot(self.A,d_0)
        self.eig=np.linalg.eigvalsh(np.dot(np.transpose(t_DA),t_DA))
        self.mineig=np.min(self.eig[self.eig>=1e-10])

    def setAdjmat(self, Adjmat, Graphtype=1, Weightype=0, SetA=False):
        self.adjmat=Adjmat
        self.graphtype=Graphtype
        self.node=np.shape(Adjmat)[0]
        self.edge=np.floor_divide(np.sum(Adjmat),2)
        
        k=0
        self.D=np.sum(Adjmat,axis=0)
        self.connect=np.zeros(self.D[self.id],dtype='i')
        for i in range(self.node):
            if(Adjmat[self.id][i]):
                self.connect[k]=i
                k=k+1
        self.setW(Weighttype=Weightype)
        if(SetA):
            self.setA()

    def setW(self, Weighttype=0):
        self.W=np.zeros(self.node)
        if(Weighttype==1):
            for j in self.connect:
                self.W[j]=1.0/(max(self.D[self.id],self.D[j]))
            self.W[self.id]=1.0-np.sum(self.W)
        elif(Weighttype==0):
            for j in self.connect:
                self.W[j]=1.0/(self.D[self.id]+1)
            self.W[self.id]=1.0-np.sum(self.W)
        else:
            for j in self.connect:
                self.W[j]=0.5/(1+max(self.D[self.id],self.D[j]))
            self.W[self.id]=1.0-np.sum(self.W)

def random_W(num_node, Nm):
    Graph = nx.Graph()
    Graph.add_nodes_from([i for i in range(num_node)])
    for i in range(num_node):
        node_list = [_ for _ in range(num_node)]
        node_list.remove(i)
        nodes = random.sample(node_list, Nm)
        edges = []
        for node in nodes:
                edges.append((i, node))
        Graph.add_edges_from(edges)
    assert nx.is_connected(Graph)
    Adj = nx.adjacency_matrix(Graph).todense()
    I = np.eye(num_node)
    DA = Adj + I
    mask = DA < 1
    random_matrix = np.random.uniform(size=(num_node, num_node))
    select = np.copy(random_matrix)
    select[mask] = .0
    P = select.T
    mode = 0
    while True:
        if mode % 2 == 0:
            P = P / np.sum(P, axis=0, keepdims=True)
        else:
            P = P / np.sum(P, axis=1, keepdims=True)
        mode = (mode+1) % 2
        if min(np.sum(P, 0)) > 1.0 -1e-7 and min(np.sum(P, 1)) > 1.0 -1e-7:
            break
    t_P = np.copy(P)
    t_P = t_P.T
    t_P[mask]=.0

    return P

def directed_W(num_node, sto_type=0, degree=3):
    # Graph = None
    # while True:
    #     Graph = nx.random_k_out_graph(num_node, num_node // 2, 1.0, self_loops=False)
    #     if nx.number_strongly_connected_components(Graph) == 1:
    #         break
    
    # Graph.add_edges_from([(i, (i+1)%num_node) for i in range(num_node)])
    # Graph.add_edges_from([(i, (i+2)%num_node) for i in range(num_node)])
    # Graph.add_edges_from([(i, (i+3)%num_node) for i in range(num_node)])
    Graph = None
    while True:
        Graph = nx.DiGraph()
        Graph.add_nodes_from([i for i in range(num_node)])
        for i in range(num_node):
            node_list = [_ for _ in range(num_node)]
            node_list.remove(i)
            nodes = random.sample(node_list, degree)
            for _ in nodes:
                Graph.add_edges_from([(i, (i+_)%num_node)])
        is_connected = nx.is_strongly_connected(Graph)
        if is_connected:
            break
    Adj = nx.adjacency_matrix(Graph).todense()
    I = np.eye(num_node)
    DA = Adj + I
    mask = DA < 1
    # random_matrix = np.random.uniform(size=(num_node, num_node))
    random_matrix = np.ones((num_node, num_node))
    select = np.copy(random_matrix)
    select[mask] = .0
    P = select.T
    if sto_type == 0: # column stochastic
        P = P / np.sum(P, axis=0, keepdims=True)
    else:
        P = P / np.sum(P, axis=1, keepdims=True)
    
    return P


def undirected_W(num_node):
    # Graph = None
    # while True:
    #     Graph = nx.random_k_out_graph(num_node, num_node // 2, 1.0, self_loops=False)
    #     if nx.number_strongly_connected_components(Graph) == 1:
    #         break
    Graph = nx.DiGraph()
    Graph.add_nodes_from([i for i in range(num_node)])
    # for _ in range(15):
    #     l = [x for x in range(100)]
    nodes = random.sample(range(30), 25)
    for _ in nodes:
        Graph.add_edges_from([(i, (i+_)%num_node) for i in range(num_node)])
        Graph.add_edges_from([((i+2)%num_node, i) for i in range(num_node)])
    assert nx.is_connected(Graph)
    Adj = nx.adjacency_matrix(Graph).todense()
    I = np.eye(num_node)
    DA = Adj + I
    mode = 0
    P = np.copy(DA)
    while True:
        if mode % 2 == 0:
            P = P / np.sum(P, axis=0, keepdims=True)
        else:
            P = P / np.sum(P, axis=1, keepdims=True)
        mode = (mode+1) % 2
        if min(np.sum(P, 0)) > 1.0 -1e-7 and min(np.sum(P, 1)) > 1.0 -1e-7:
            break
    
    return P
        
