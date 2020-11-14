import snap


def create_graph():
    g = snap.TUNGraph.New()
    for i in range(1, 11):
        g.AddNode(i)

    g.AddEdge(1, 2)
    g.AddEdge(1, 3)
    g.AddEdge(2, 3)
    g.AddEdge(2, 4)
    g.AddEdge(3, 6)
    g.AddEdge(4, 7)
    g.AddEdge(4, 8)
    g.AddEdge(5, 8)
    g.AddEdge(5, 6)
    g.AddEdge(5, 9)
    g.AddEdge(6, 9)
    g.AddEdge(6, 10)
    g.AddEdge(7, 8)
    g.AddEdge(8, 9)
    g.AddEdge(9, 10)
    return g


G = create_graph()
node_dict = {}
positive = [1, 0]
negative = [0, 1]
node_dict[3] = positive
node_dict[5] = positive
node_dict[8] = negative
node_dict[10] = negative
label_id = [3, 5, 8, 10]
node_num = G.GetNodes()

for i in range(1, node_num + 1):
    if i not in label_id:
        node_dict[i] = [0.5, 0.5]

flag = 1
loop_cnt = 0
while flag is not 0:
    # 当不在变化时，停止迭代
    flag = 0
    for i in range(1, node_num + 1):
        if i not in label_id:
            neighbors = []
            cur_node = G.GetNI(i)
            degree = cur_node.GetDeg()

            for nbr in range(degree):
                neighbors.append(cur_node.GetNbrNId(nbr))
            origin = node_dict[i]
            sum_p = [0, 0]
            for mid in neighbors:
                sum_p[0] += node_dict[mid][0] / degree
                sum_p[1] += node_dict[mid][1] / degree
            node_dict[i] = sum_p
            if abs(origin[0] - sum_p[0]) > 0.001:
                # 当每次变化小于0.001时，认为仍在变化
                flag += 1
            print('id:{}\t pro:{}'.format(i, sum_p))
    loop_cnt += 1
    print('Loop {} finish!!!'.format(loop_cnt))
