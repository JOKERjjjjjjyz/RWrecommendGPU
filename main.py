import dataloader
import world
import torch
from dataloader import Loader
import sys
import scipy.sparse as sp
from train import *
import numpy as np
from scipy.sparse import csr_matrix


if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="./data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.Loader(path="./data")

num_rows, num_cols = dataset.UserItemNet.shape
vector_origin = []

# 遍历每一行
for row_idx in range(num_rows):
    # 获取当前行的起始和结束索引
    start_idx = dataset.UserItemNet.indptr[row_idx]
    end_idx = dataset.UserItemNet.indptr[row_idx + 1]

    # 获取当前行的列索引和对应的非零元素
    row_indices = dataset.UserItemNet.indices[start_idx:end_idx]
    row_data = dataset.UserItemNet.data[start_idx:end_idx]

    # 初始化一个零向量
    row_vector = np.zeros(num_cols)

    # 将非零元素赋值给向量的相应位置
    for col_idx, value in zip(row_indices, row_data):
        row_vector[col_idx] = value

    # 将当前行向量添加到向量数组中
    vector_origin.append(row_vector)

graph = dataset.getSparseGraph()
graph = graph.tocsr()

# M:user number; N: item number
# vector_origin: M*N;  vector_propagate: (M+N)*N
index = world.seed
M = dataset.n_users
N = dataset.m_items
K_value = eval(world.topks)
K = K_value[0]
vector_propagate = [np.zeros((M + N, N)) for _ in range(K)]
# vector_propagate_sum = np.zeros((M + N, N))  # 创建用于存储总和的矩阵
graph_dense = graph.toarray()

graph = torch.tensor(graph_dense, dtype=torch.float32, device='cuda')
user_vectors = [torch.tensor(vec, device='cuda') for vec in vector_origin]
vector_origin = torch.stack(user_vectors, dim=0)  # 批量维度是 0
vector_propagate_sum = torch.zeros((M + N, N), dtype=torch.float32, device='cuda')

testarray = [[] for _ in range(M)]
for idx, user in enumerate(dataset.test):
    testarray[idx] = dataset.test[user]
test_indices = []  # 初始化一个空列表用于存储测试物品的索引

# 遍历 testarray 中的每个子列表
for user_items in testarray:
    # 将用户的测试物品索引添加到 test_indices 中
    test_indices.extend(user_items)

# 将 test_indices 转换为 PyTorch 张量并移动到 GPU
test = torch.tensor(test_indices, dtype=torch.long, device='cuda')

# 现在，test 是一个包含所有测试物品索引的 PyTorch 张量，可以在后续的操作中使用

for i in range(1,K+1):
    sampleNum = Klayer_sampleNum(i,0.025, 0.5, M,index)
    print(type(vector_origin))
    vector_propagate[i-1] = propagateGpu(i,graph,vector_origin,M,N,sampleNum)
    updated_vector = vector_propagate.cpu().numpy()
    filename = f"{world.dataset}_matrix_{i-1}.npy"  # 文件名类似于 matrix_0.npy, matrix_1.npy, ...
    np.save(filename, updated_vector[i-1])
    vector_propagate_sum += vector_propagate[i-1]
    recommendList, recommend_vector = topKGpu(vector_origin, vector_propagate_sum, M, N, 20)
    count = evaluateGpu(recommendList, test)
    recall = count / dataset.testDataSize
    print("epoch:",i," recall:", recall)

updated_sumvector = vector_propagate_sum.cpu().numpy()
filename = f"matrix_sum.npy"  # 文件名类似于 matrix_0.npy, matrix_1.npy, ...
np.save(filename, updated_sumvector)
original_stdout = sys.stdout
with open('recall_output.txt', 'w') as f:
    # 重定向 stdout 到文件
    sys.stdout = f
    print("Final recall:", recall)
sys.stdout = original_stdout