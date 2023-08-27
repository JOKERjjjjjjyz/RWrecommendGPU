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

# 将向量数组转换为 NumPy 数组
vector_array = np.array(vector_origin)

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
vector_propagate_sum = np.zeros((M + N, N))  # 创建用于存储总和的矩阵
testarray = [[] for _ in range(M)]
for idx, user in enumerate(dataset.test):
    testarray[idx] = dataset.test[user]

for i in range(1,K+1):
    sampleNum = Klayer_sampleNum(i,0.025, 0.5, M,index)
    vector_propagate[i-1] = propagate(i,graph,vector_origin,M,N,sampleNum)
    filename = f"{world.dataset}_matrix_{i-1}.npy"  # 文件名类似于 matrix_0.npy, matrix_1.npy, ...
    np.save(filename, vector_propagate[i-1])
    vector_propagate_sum += vector_propagate[i-1]
    recommendList, recommend_vector = topK(vector_origin, vector_propagate_sum, M, N, 20)
    count = evaluate(recommendList, testarray)
    recall = count / dataset.testDataSize
    print("epoch:",i," recall:", recall)

filename = f"matrix_sum.npy"  # 文件名类似于 matrix_0.npy, matrix_1.npy, ...
np.save(filename, vector_propagate_sum)

recommendList,recommend_vector = topK(vector_origin,vector_propagate_sum,M,N,20)
recommend_vector_csr = csr_matrix(recommend_vector)
sp.save_npz(dataset.path + '/recommend_vector.npz', recommend_vector_csr)
count = evaluate(recommendList , testarray)
recall = count / dataset.testDataSize
print ("Final recall:",recall)
# dense_array = dataset.UserItemNet.toarray()

# 将原始 stdout 保存到变量
original_stdout = sys.stdout

# 打开一个文件来替代 stdout
with open('recall_output.txt', 'w') as f:
    # 重定向 stdout 到文件
    sys.stdout = f
    print("Final recall:", recall)
    # 现在所有的 print 输出都会写入到文件中
    # with np.printoptions(threshold=np.inf):
    #     print("0:",vector_array[0])
    #     print("1:",vector_array[1])
    # print("users:",dataset.n_users)
    # print("items:",dataset.m_items)
# 恢复原始的 stdout
sys.stdout = original_stdout