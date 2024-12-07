import argparse
import numpy as np
from scipy.optimize import linprog
import os
import time
import json

np.seterr(divide='ignore', invalid='ignore')
def wassersteindistance(a, b, c):
    A_eq = []
    for i in range(len(a)):
        A = np.zeros_like(c)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(b)):
        A = np.zeros_like(c)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    b_eq = np.concatenate([a, b])
    "c:infinte"
    c = c.reshape(-1)
    #print(np.isfinite(c).any())
    result = linprog(c, A_eq=A_eq, b_eq=b_eq)
    result = result.fun
    return result

def spatial_temporal_aware_distance(x, y):
    x_norm = (x ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_norm = (y ** 2).sum(axis=1, keepdims=True) ** 0.5

    a = x_norm[:, 0] / x_norm.sum()
    b = y_norm[:, 0] / y_norm.sum()

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_mean_norm = ((x - x_mean) ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_mean_norm = ((y - y_mean) ** 2).sum(axis=1, keepdims=True) ** 0.5
    c = np.dot((x - x_mean) / x_mean_norm, ((y - y_mean) / y_mean_norm).T)
    c = np.nan_to_num(c)
    D = -wassersteindistance(a, b, c)
    return D

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/PEMS07.json')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = json.loads(f.read())
data = config['data']
dataset = config['dataset']
day = config['day']
threshold = config['threshold']
filename = data + '/' + dataset + '/' + dataset + '.npz'
all_data = np.load(filename)['data']
num_of_samples, nodes, _ = all_data.shape
train_t = int(num_of_samples * 0.6)
train_t = int(train_t / day)*day
all_data = all_data[:train_t, :, :1].reshape([-1, day, nodes])


d = np.zeros([nodes, nodes])
for i in range(nodes):
    t1 = time.time()
    for j in range(i+1, nodes):
        d[i, j] = spatial_temporal_aware_distance(all_data[:, :, i], all_data[:, :, j])
        if d[i, j] < threshold:
            d[i, j] = 1
        else:
            d[i, j] = 0
    t2 = time.time()
    print('sum_epoch is', nodes, ',current epoch is', i+1, ',finished waste time is', t2-t1)

d = d+d.T
d = d + np.identity(nodes)
STAM_filename = data + '/' + dataset + '/' + dataset + '_STAM'+'.npy'
if os.path.exists(STAM_filename):
    os.remove(STAM_filename)
    np.save(STAM_filename, d)
else:
    np.save(STAM_filename, d)
print("STAM is generated!")






