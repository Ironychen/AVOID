import re
from networkx.algorithms.shortest_paths import weighted
from networkx.classes.function import is_directed
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import networkx as nx
# import dgl

import torch
from torch_geometric.data import Data



# Get the tkoen ids of sentences 将自然语言转token_id
def convert_text_to_token(tokenizer, new, limit_sens, limit_words):
    result_new = []
    # new = new.split(".?!")
    new = re.split(r'[.?!;]',new)
    # print(len(new))
    # print(new)
    new = new[:-1] # drop the last full stop 去掉最后的句号
    nodes_num = limit_sens
    for sen in new: # get the tkoen id of every sentece 每一句提取词tokenid
        # print(sen)
        if len(sen.split(" ")) < 3:
            # print("1\n")
            pass
        else:
            tokens = tokenizer.encode(sen, max_length=limit_words, padding="max_length", truncation=True, return_tensors='pt') # token ids of one sentence's word 一句话的tokenid输入
            result_new.append(tokens) # 添加tokenid
            if len(result_new) >= limit_sens: 
                break
    if len(result_new) < limit_sens: # if less than setting senteces number, we padding it 如果少于设置的超参数句子数量，则补全
        nodes_num = len(result_new)
        need_pad = limit_sens - len(result_new)
        result_new.extend([torch.Tensor([[0] * limit_words]) for i in range(need_pad)])
    result_new = torch.stack(result_new, 0)
    result_new = torch.squeeze(result_new, 1)
    # return result_new
    return result_new, nodes_num

# Reduce the dimesion of Bert features
def lower_dimension(feature, hidden_dim):
    if hidden_dim == 768:
        pass
    else:
        # tsne_feature = TSNE(n_components=hidden_dim, init='pca', random_state=100,early_exaggeration=15,perplexity=25)
        # tsne_feature = tsne_feature.fit_transform(feature.cpu())
        tsne_feature = PCA(n_components=hidden_dim, svd_solver='auto')
        tsne_feature = tsne_feature.fit_transform(feature.cpu())

        print("Org data dimension is {}. Embedded data dimension is {}".format(feature.shape[-1], tsne_feature.shape[-1]))

# Build a graph
def build_graph(nodes_feature):
    # print(nodes_feature)
    # nodes_feature = nodes_feature[0:node_num]
    sim = []
    for i in nodes_feature:
        for j in nodes_feature:
            sim_sent = torch.abs(torch.cosine_similarity(i,j,dim=0))
            sim.append(sim_sent.item())
    # print(len(nodes_feature[1]))
    # if(nodes_feature[1][-1][0]==0 and nodes_feature[1][-1][-1]==0):
        # nodes_feature = nodes_feature[:,:-1,:]
    node_num = len(nodes_feature)
    
    # print(node_num)
    # print(nodes_feature.shape)
    
    G = nx.Graph()
    edge_list = []
    # weight_list = []
    for i in range(node_num):
        # edge_list.append((i,i))
        for j in range(i, node_num):
            edge_list.append((i,j)) # Full connect will lead to all nodes' features are the same. 全连接导致所有节点的特征在图卷积聚合一次过后是相同的，这里可以考虑一下后续的更新
            # weight_list.append(0.5)
    # print(nodes_feature)
    # edgelist = [(0, 1), (0, 2), (1, 3)]  # note that the order of edges
    # print(edge_list)
    # G.add_edges_from(edge_list)
    G.add_edges_from(edge_list)
    # print(nx.adj_matrix(G).todense())
    # for i in range(node_num):
    #     x[i][i] = nodes_feature[i]

    adj = nx.to_scipy_sparse_array(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.tensor(sim, dtype=torch.float32)

    data = Data(x=nodes_feature, edge_index=edge_index, edge_attr=edge_weight)
    # data = Data(x=nodes_feature, edge_index=edge_index)
    return data

# Get data from the dataset
def get_data(data_path):
    news_list = pd.read_csv(data_path, sep="\t")
    # news_list = shuffle(news_list) # shuffle the dataset 打乱数据集
    # print(news_list)

    # Get the title, content, and label of news, respectively. 分别获取新闻的标题、内容、标签。
    news_title, news_content, news_label, news_id = [],[],[],[]
    count = 0
    # list_name = ["clean_title","content","2_way_label","id"]
    list_name = ["clean_title","content","2_way_label","id"]
    for names_index in range(len(list_name)):
        for news_attr in news_list[list_name[names_index]]:
            # print(f"this is one of the news attributes: {news_attr}")        
            if count == 0:
                news_title.append(news_attr)
            if count == 1:
                news_content.append(news_attr)
            if count == 2:
                news_label.append(news_attr)
            if count == 3:
                news_id.append(news_attr)
        count += 1

    # Merge the news title and content, if necessary. 若有必要，将标题和内容合并为一个段落。
    news_tit_cont = []
    print(f"the length of news_list is: {len(news_list)}")
    for i, tit in enumerate(news_title):
        news_tit_cont.append(f"{tit}. {news_content[i]}")
    print(f"the length of news_tit_cont is: {len(news_tit_cont)}")
    # print(news_id)
    return news_tit_cont, news_label, news_id
    # return news_tit_cont, news_label




