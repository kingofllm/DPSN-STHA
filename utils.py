import numpy as np
from scipy.sparse.linalg import eigs
'''
def adj_weight_matrix(datadir, dataset, nodes, type):
    #df = pd.read_csv(datadir + '/' + dataset + '/' + dataset + '.csv')
    #df = df.to_numpy()
    #id_filename = datadir + '/' + dataset + '/' + dataset + '.txt'
    #adj = np.zeros((nodes, nodes), dtype=np.float32)
    """
    with open(id_filename, 'r') as f:
        id = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
    print(id)
    """
    filename = datadir + '/' + dataset + '/' + dataset + '.csv'
    df = pd.read_csv(filename)
    df = df.to_numpy()
    A = np.zeros((nodes, nodes), dtype=float)
    for row in range(df.shape[0]):
        i, j, distance = int(df[row][0]), int(df[row][1]), float(df[row][2])
        if type == 'connectivity':
            A[i, j] = 1
            A[j, i] = 1
        elif type == 'distance':
            A[i, j] = 1 / distance
            A[j, i] = 1 / distance
        else:
            raise ValueError("type must be connectivity or distance!")
    return A
'''
def calculate_graph_laplacian_matrix(STAM, nodes):
    D = np.zeros((nodes, nodes))
    for i in range(nodes):
       D[i, i] = np.sum(STAM[i, :])
    graph_lap = D - STAM
    return graph_lap

def calculate_cheb_laplacian_matrix(graph_lap, nodes):
    #返回k个最大特征值
    feature_value_max = eigs(graph_lap, k=1, which='LR')[0].real
    N = np.identity(nodes)
    cheb_lap = (2 * graph_lap)/feature_value_max - N
    return cheb_lap

def cheb_polynomial(cheb_lap, m):
    N = cheb_lap.shape[0]
    cheb_polynomials = [np.identity(N), cheb_lap]
    for i in range(2, m):
        cheb_polynomials.append(2 * cheb_lap * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials





