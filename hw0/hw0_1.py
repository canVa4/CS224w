import snap
import numpy as np
import matplotlib.pyplot as plt

data_dir = './data/Wiki-Vote.txt'
WikiG = snap.LoadEdgeList(snap.PNGraph, data_dir, 0, 1, '\t')


def partOneAndTwo(WikiG):
    # WikiG.Dump()
    print('1. Number of nodes: '+str(WikiG.GetNodes()))

    selfloop_cnt = 0
    for node in WikiG.Nodes():
        # print(node.GetId())
        if WikiG.IsEdge(node.GetId(), node.GetId()):
            selfloop_cnt += 1
    print('2. Self loop Node: {}'.format(selfloop_cnt))

    cnt_dir = snap.CntUniqDirEdges(WikiG)
    print('3. The number of directed edges: {}'.format(cnt_dir))

    cnt_undir = snap.CntUniqUndirEdges(WikiG)
    print("4. The number of unique undirected edges is %d" % cnt_undir)
    print("5. The number of reciprocated edges is %d" % (cnt_dir - cnt_undir))

    cnt_in = snap.CntInDegNodes(WikiG, 0)
    print("6. The number of nodes of zero out-degree is %d" % cnt_in)
    cnt_out = snap.CntOutDegNodes(WikiG, 0)
    print("7. The number of nodes of zero in-degree is %d" % cnt_out)

    cnt_deg_above_10 = 0
    cnt_deg_less_10 = 0
    for node in WikiG.Nodes():
        if node.GetOutDeg() > 10:
            cnt_deg_above_10 += 1
        if node.GetInDeg() < 10:
            cnt_deg_less_10 += 1
    print("8. The number of nodes with more than 10 outgoing edges is %d" % cnt_deg_above_10)
    print("9. The number of nodes with fewer than 10 incoming edges is %d" % cnt_deg_less_10)


    # Part 2
    out_file_name = 'wiki'
    snap.PlotInDegDistr(WikiG, out_file_name, "Directed graph - in-degree Distribution")
    snap.PlotOutDegDistr(WikiG, out_file_name, "Directed graph - out-degree Distribution")

    InDegDistr = np.loadtxt("inDeg."+out_file_name+".tab")
    InDegDistr = InDegDistr[InDegDistr[:, 0] > 0]

    OutDegDistr = np.loadtxt("OutDeg."+out_file_name+".tab")
    # print(OutDegDistr.shape)
    OutDegDistr = OutDegDistr[OutDegDistr[:, 0] > 0]
    # print(OutDegDistr.shape)

    coff = np.polyfit(np.log10(OutDegDistr)[:, 0], np.log10(OutDegDistr)[:, 1], 1)
    print(coff)
    plt.figure()
    plt.subplot(211)
    plt.loglog(InDegDistr[:, 0], InDegDistr[:, 1])
    plt.title('In deg Distr')
    plt.subplot(212)
    plt.loglog(OutDegDistr[:, 0], OutDegDistr[:, 1])
    plt.loglog(OutDegDistr[:, 0], np.power(10, coff[1])*np.power(OutDegDistr[:, 0], coff[0]))
    plt.title('Out deg Distr & Last-Square Reg Line in log-log plot')
    plt.show()


def partThree():
    data_dir_StackOverFlow = './data/stackoverflow-Java.txt'
    sofG = snap.LoadEdgeList(snap.PNGraph, data_dir_StackOverFlow, 0, 1, '\t')

    Components = snap.TCnComV()
    snap.GetWccs(sofG, Components)
    print('1. The number of weakly connected components in the network.: '+str(Components.Len()))

    MxWcc = snap.GetMxWcc(sofG)
    num_node = MxWcc.GetNodes()
    num_deg = MxWcc.GetEdges()
    print('2. The number of edges is {} and the number of nodes is {}'.format(num_deg, num_node))

    PRankH = snap.TIntFltH()
    snap.GetPageRank(sofG, PRankH)
    cnt = 0
    print('3. ')
    for item in PRankH:
        cnt += 1
        if cnt > 3:
            break
        print(item, PRankH[item])

    print('4. ')
    NIdHubH = snap.TIntFltH()
    NIdAuthH = snap.TIntFltH()
    snap.GetHits(sofG, NIdHubH, NIdAuthH)
    HubDict = {}
    AuthDict = {}
    for item in NIdHubH:
        HubDict[item] = NIdHubH[item]
    a = zip(HubDict.values(), HubDict.keys())
    print(list(sorted(a, reverse=True))[:3])
    for item in NIdAuthH:
        AuthDict[item] = NIdAuthH[item]
    b = zip(AuthDict.values(), AuthDict.keys())
    print(list(sorted(b, reverse=True))[:3])


partOneAndTwo(WikiG)
partThree()
