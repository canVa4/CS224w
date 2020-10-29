import snap
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def load_binary_graph():
    """
    读取二进制存储的图
    :return: SNAP中的 TUNGraph
    """
    G = snap.TUNGraph.Load(snap.TFIn("hw1-q2.graph"))
    return G


def extract_basic_features(node, graph):
    """
    提取basic features
    :param node: 目标节点，SNAP中的node类，而非ID，可以用GetNI()获得
    :param graph: 目标图（无向图）
    :return: 长度为3的array 分别为：该节点deg，egonet内部deg，进出egonet的边数
    """
    degree = node.GetDeg()

    neighbors = []

    # 计算全部邻居的deg的和
    total_deg_nbr = 0
    for i in range(degree):
        # 获取全部的邻居，目标构建egonet
        neighbor = graph.GetNI(node.GetNbrNId(i))
        neighbors.append(neighbor)
        total_deg_nbr += neighbor.GetDeg()

    # 计算邻居之间边的数量
    edge_between_nbr = 0
    for i in range(len(neighbors)):
        for j in range(i):
            edge_between_nbr += neighbors[i].IsNbrNId(neighbors[j].GetId())

    return np.array((degree, edge_between_nbr + degree, total_deg_nbr - 2 * edge_between_nbr - degree))


def extract_basic_feature_others(node, graph):
    v = [node.GetDeg()]
    tot_edges = 0
    nbrs = []
    for i in range(v[0]):
        nbrs.append(graph.GetNI(node.GetNbrNId(i)))
        tot_edges += nbrs[-1].GetDeg()
    inner_edges = 0
    for i in range(v[0]):
        for j in range(i):
            inner_edges += nbrs[i].IsInNId(nbrs[j].GetId())
    v.append(inner_edges)
    v.append(tot_edges - 2 * inner_edges)

    return np.array(v)


def cosine_similarity(x, y):
    """
    计算余弦相似度 x与y为两个等长array
    :param x: array
    :param y: array
    :return: 余弦相似度
    """
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0
    return np.dot(x, y) / (norm_x * norm_y)


def CalculateLocalVector(Graph):
    # dim-1 : the degree of v,
    # dim-2 : the number of edges in the egonet of v
    # dim-3 : the number of edges that connect the egonet of v and the rest of the graph,
    #           i.e., the number of edges that enter or leave the egonet of v.

    N = Graph.GetNodes()
    V_local = np.zeros((N, 3))

    idx = 0
    for NI in Graph.Nodes():
        V_local[idx, 0] = NI.GetDeg()
        # print(idx, NI.GetId()) # they are equal

        V_ids = snap.TIntV()
        V_ids.Add(NI.GetId())
        degree_sum = NI.GetDeg()
        for Id in NI.GetOutEdges():
            V_ids.Add(Id)
            degree_sum += Graph.GetNI(Id).GetDeg()
            # print("edge (%d %d)" % (NI.GetId(), Id))
        G_ego = snap.ConvertSubGraph(snap.PUNGraph, Graph, V_ids)

        V_local[idx, 1] = G_ego.GetEdges()
        V_local[idx, 2] = degree_sum - 2 * G_ego.GetEdges()

        idx += 1
    return V_local


def q2_1():
    print("Answer for Q2.1:")
    G = load_binary_graph()
    basic_mat = []
    for node in G.Nodes():
        basic_mat.append(extract_basic_features(node, G))
    basic_mat = np.array(basic_mat)
    basic_v9 = basic_mat[9]
    print(basic_v9)
    res = []
    test = []
    for i in range(basic_mat.shape[0]):
        cossim = cosine_similarity(basic_mat[i], basic_v9)
        res.append((i, cossim))
    sorted_res = sorted(res, key=lambda k: k[1], reverse=True)
    print('Top 5:')
    for id, cossim in sorted_res[1:6]:
        print('id:{}\t cos sim:{}'.format(id, cossim))
    print("Finish Q2.1\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


def recursive_features(v_mat, graph):
    """
    迭代一次
    :param v_mat: np array
    :param graph: 目标图
    :return: 新的v_mat，向量长度变为原来的3倍
    """
    cnt_node, ori_len = v_mat.shape
    new_mat = np.zeros((cnt_node, ori_len))
    mean_mat = np.zeros((cnt_node, ori_len))
    sum_mat = np.zeros((cnt_node, ori_len))
    for node in graph.Nodes():
        nodeId = node.GetId()
        cnt_nbrs = node.GetDeg()
        new_mat[nodeId] = v_mat[nodeId]
        if cnt_nbrs == 0:
            continue
        for i in range(cnt_nbrs):
            nbr_id = node.GetNbrNId(i)
            mean_mat[nodeId] += v_mat[nbr_id]
            sum_mat[nodeId] += v_mat[nbr_id]
        mean_mat[nodeId] = mean_mat[nodeId] / cnt_nbrs
    return np.concatenate((new_mat, mean_mat, sum_mat), axis=1)


def q2_2():
    K = 2
    print("Answer for Q2.2:")
    G = load_binary_graph()
    feature_mat = []
    for node in G.Nodes():
        feature_mat.append(extract_basic_features(node, G))
    feature_mat = np.array(feature_mat)

    for i in range(K):
        feature_mat = recursive_features(feature_mat, G)

    basic_v9 = feature_mat[9]
    res = []
    for i in range(feature_mat.shape[0]):
        cossim = cosine_similarity(feature_mat[i], basic_v9)
        res.append((i, cossim))
    sorted_res = sorted(res, key=lambda k: k[1], reverse=True)
    print('Top 5:')
    for id, cossim in sorted_res[1:6]:
        print('id:{}\t cos sim:{}'.format(id, cossim))
    print("Finish Q2.2\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


def q2_3():
    K = 2
    print("Answer for Q2.3:")
    G = load_binary_graph()
    feature_mat = []
    for node in G.Nodes():
        feature_mat.append(extract_basic_features(node, G))
    feature_mat = np.array(feature_mat)
    for i in range(K):
        feature_mat = recursive_features(feature_mat, G)

    basic_v9 = feature_mat[9]
    res = []
    for i in range(feature_mat.shape[0]):
        cossim = cosine_similarity(feature_mat[i], basic_v9)
        res.append(cossim)
    # print(res)
    res = np.array(res)
    plt.hist(x=res, bins=20, range=(0, 1))
    plt.ylabel('node counts')
    plt.xlabel('cosine similarity with ID9')
    plt.show()


def Draw_SubGraph(Graph, Node_id):
    """
    绘图函数 Node 节点2-hop子图
    :param Graph: 目标图
    :param NI_id: 目标节点
    :return: 无
    """
    Nodes = []
    Edges = []
    center_node = Graph.GetNI(Node_id)
    Nodes.append(Node_id)
    colors = []

    # 所有1-hop节点放进去
    for i in range(center_node.GetDeg()):
        nbr_id = center_node.GetNbrNId(i)
        Nodes.append(nbr_id)
    # 所有2-hop节点放进去
    for i in range(len(Nodes)):
        mid_node = Graph.GetNI(Nodes[i])
        for j in range(mid_node.GetDeg()):
            nbr_id = mid_node.GetNbrNId(j)
            Nodes.append(nbr_id)
    Nodes = list(set(Nodes))

    for mid_id in Nodes:
        if mid_id == Node_id:
            colors.append('r')
        else:
            colors.append('b')
    # print(Nodes)
    for i in range(len(Nodes)):
        for j in range(i):
            if Graph.IsEdge(Nodes[i], Nodes[j]):
                Edges.append((Nodes[i], Nodes[j]))
    # print(len(Edges))
    G = nx.Graph()
    G.add_nodes_from(Nodes)
    G.add_edges_from(Edges)
    nx.draw_networkx(G, node_color=colors)
    # plt.show()


def q2_3_plot():
    G = load_binary_graph()
    # Plot test
    # plt.figure()
    # plt.subplot(121)
    # Draw_SubGraph(G, 9)
    # plt.subplot(122)
    # Draw_SubGraph(G, 100)
    # plt.show()

    K = 2
    print("Answer for Q2.4:")
    feature_mat = []
    for node in G.Nodes():
        feature_mat.append(extract_basic_features(node, G))
    feature_mat = np.array(feature_mat)
    for i in range(K):
        feature_mat = recursive_features(feature_mat, G)

    basic_v9 = feature_mat[9]
    res = []
    for i in range(feature_mat.shape[0]):
        cossim = cosine_similarity(feature_mat[i], basic_v9)
        res.append(cossim)

    control = [0, 0, 0]
    bias = 500
    res = res[bias:-1]
    chosen_dict = {}
    for i, cossim in enumerate(res):
        if control[0] == 0 and cossim > 0.6 and cossim < 0.65:
            control[0] = 1
            chosen_dict[0] = i
        if control[1] == 0 and cossim > 0.85 and cossim < 0.9:
            control[1] = 1
            chosen_dict[1] = i
        if control[2] == 0 and cossim > 0.9 and cossim < 0.95:
            control[2] = 1
            chosen_dict[2] = i
        if control[0] + control[1] + control[2] == 3:
            break
    print(chosen_dict)
    plt.figure()
    plt.subplot(131)
    Draw_SubGraph(G, chosen_dict[0] + bias)
    plt.subplot(132)
    Draw_SubGraph(G, chosen_dict[1] + bias)
    plt.subplot(133)
    Draw_SubGraph(G, chosen_dict[2] + bias)

    plt.show()


# q2_1()
# q2_2()
q2_3()
q2_3_plot()
