import numpy as np
import random
import math
import torch
def randomwalk(length, graph, start_node):
    current_node = start_node
    for step in range(length):
        # 获取当前节点的邻居节点
        neighbors = torch.nonzero(graph[current_node]).flatten()
        if len(neighbors) == 0:
            # 当前节点没有邻居，随机游走结束
            break
        # 随机选择一个邻居作为下一步的节点
        next_node = torch.randint(len(neighbors), size=(1,), device='cuda')
        next_node = neighbors[next_node]
        # 更新当前节点为下一步的节点
        current_node = next_node
    radio = 1 / length
    return current_node, radio

def propagateGpu(k,graph,vector_origin,M,N,KsampleNum):
    vector = torch.zeros((M + N, N), dtype=torch.float32, device='cuda')
    for user_idx in range(M):
        for j in range(KsampleNum):
            print("Training:Epoch",k,",(user,j):(",user_idx,",",j,")")
            targetNode,radio = randomwalk(k, graph, user_idx)
            vector[targetNode] += radio*vector_origin[user_idx]*0.001
    return vector

def Klayer_sampleNum(k,epsilon,delta,M,index):
    # return N: sample number for k
    N = 1/(2*epsilon*epsilon)*math.log(2*M/delta)*M*math.pow(k,index)
    return int(N)+1

def topKGpu(vector_origin,vector_propagate,M,N,k):
    recommendList = []
    recommend_vector = [torch.zeros(N, dtype=torch.float32, device='cuda') for _ in range(M)]
    for user in range(M):
        for j in range(N):
            if vector_origin[user][j] != 0 :
                vector_propagate[user][j] = 0;
        sorted_indices = torch.argsort(vector_propagate[user])
        topk_indices = sorted_indices[-k:]
        recommend_vector[user][topk_indices] = 1
        user_recommendList = topk_indices.tolist()
        user_recommendList = [(user, idx) for idx in user_recommendList]
        recommendList.extend(user_recommendList)
    return recommendList, recommend_vector

def evaluateGpu(recommendList, test):
    count = torch.tensor(0, dtype=torch.int32, device='cuda')
    print("Evaluating...")
    for tuple_item in recommendList:
        user = tuple_item[0]
        item = tuple_item[1]
        test_user = torch.tensor(test[user], dtype=torch.int64, device='cuda')
        if torch.any(test_user == item):
            count += 1
    return count.item()