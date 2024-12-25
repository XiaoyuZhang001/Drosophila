# 这是一个示例 Python 脚本。
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import scipy.stats as stats
import random
from scipy.spatial import cKDTree
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
# class Neutron(object):  # 创建Circle类
#    def __init__(self,x,y,z,next = None):
#        self.x = x
#        self.y = y
#        self.z = z
#        self.next = next
def fit_distribution(data):
    # 定义要拟合的分布
    distributions = {
        'normal': stats.norm,
        'exponential': stats.expon,
        'uniform': stats.uniform
    }

    best_fit_name = None
    best_fit_params = None
    lowest_aic = np.inf

    for name, dist in distributions.items():
        # 拟合分布并获取参数
        params = dist.fit(data)

        # 使用最大似然估计的负值作为AIC的一部分
        mle = dist.nnlf(params, data)
        # 计算参数的数量
        k = len(params)
        # 计算AIC
        aic = 2 * k + 2 * mle

        if aic < lowest_aic:
            best_fit_name = name
            best_fit_params = params
            lowest_aic = aic
    with open('distribution.txt', 'a') as f:
        # print(source + '-' + target + '=', end='', file=f)
        # print(sum(length) / len(length), file=f)
        print(f"Best fitting distribution: {best_fit_name}", file=f)
        print(f"Parameters for the best fit: {best_fit_params}", file=f)

    return best_fit_name, best_fit_params
def load_data(path="synapse_coordinates.csv"):
    data = pd.read_csv(path)
    # print(data)
    array = data.values[0::, 0::]  # 读取全部行，全部列
    # array = array[:, 0:2]
    # print(np.isnan(array[1,1]))
    # print(int(array[0][0]))  # array是数组形式存储，顺序与data读取的数据顺序格式相同
    # print(str(int(int(array[0][0]) / 1000) * 1000))
    for i in range(len(array)):
        if not np.isnan(array[i][0]):
            # array[i][0] = int(int(array[i][0])/1000)*1000
            temp1 = array[i][0]
            num = 0
        else:
            array[i][0] = temp1

    for i in range(len(array)):
        if not np.isnan(array[i][1]):
            # array[i][1] = int(int(array[i][1]) / 1000) * 1000
            temp1 = array[i][1]
            num = 0
        else:
            array[i][1] = temp1
    print("load_done")
    return array


def process_data():
    return


def load_synapse():
    # 在下面的代码行中使用断点来调试脚本。
    node_pair1 = np.loadtxt('./synapse_coordinates.txt', dtype=bytes)
    node_pair = []
    for item in node_pair1:
        node_pair.append(str(item)[2:-1].split(','))
    # print(node_pair[0:50])

    # print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
    return node_pair


def load_position():
    # node_pair1 = np.loadtxt('./coordinates.txt', dtype=bytes)
    pos_pair = []
    with open('./coordinates.txt', "rb") as f:
        i = 0
        for line in f.readlines():  # 打开后逐行读取
            each_line = line.decode().rstrip().split(',')
            # each_line[1] = each_line[1].split(' ')
            each_line1 = []
            # if i == 27833:
            #     print(each_line)
            #     print(each_line[1].split())
            each_line1.append(each_line[0])
            # 防止位数不对现象导致数据处理错误
            if len(each_line[1].split()) == 3:
                each_line1.append(each_line[1].split()[0][1:])
                each_line1.append(each_line[1].split()[1])
                each_line1.append(each_line[1].split()[2][:-1])
            elif len(each_line[1].split()) == 4:
                each_line1.append(each_line[1].split()[1])
                each_line1.append(each_line[1].split()[2])
                each_line1.append(each_line[1].split()[3][:-1])
            # print(each_line[1])
            pos_pair.append(each_line1)
            i += 1
    # node_pair = node_pair[1:]
    # print(pos_pair[10])
    # print(len(pos_pair[10]))
    return pos_pair


def load_classification():
    # node_pair1 = np.loadtxt('./coordinates.txt', dtype=bytes)
    class_pair = []
    with open('./classification.txt', "rb") as f:
        for line in f.readlines():  # 打开后逐行读取
            each_line = line.decode().rstrip().split(',')
            # print(each_line)
            # each_line[1] = each_line[1].split(' ')
            class_pair.append(each_line)
    # node_pair = node_pair[1:]
    # print(pos_pair[10])
    # print(len(pos_pair[10]))
    return class_pair


def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as load_f:
        str_f = load_f.read()
        if len(str_f) > 0:
            datas = json.loads(str_f)
        else:
            datas = {}
    return datas
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def shuffle_G(G):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    # 创建一个新的图
    G_shuffled = nx.DiGraph()
    # 添加打乱后的节点和边，并复制属性
    node_mapping = {old_label: new_label for old_label, new_label in zip(G.nodes(), nodes)}
    for u, v, data in G.edges(data=True):
        G_shuffled.add_edge(node_mapping[u], node_mapping[v], **data)

    # 复制节点属性
    for node, data in G.nodes(data=True):
        G_shuffled.add_node(node_mapping[node], **data)
    return G_shuffled
# 按间距中的绿色按钮以运行脚本。
def preprocess_graph(G):
    nodes_to_remove = []
    for node, data in G.nodes(data=True):
        if 'direction' not in data or not isinstance(data['direction'], list) or len(data['direction']) != 3:
            # print(f"Node {node} has invalid or missing direction: {data.get('direction', 'N/A')}")
            nodes_to_remove.append(node)
        else:
            # print(f"Node {node} has valid direction: {data['direction']}")
            pass
    print("Number of nodes to remove:", len(nodes_to_remove))
    print("Nodes to remove:", nodes_to_remove)

    G.remove_nodes_from(nodes_to_remove)
    return G

# def remove_sphere_nodes(G, center, radius):
#     # 提取所有节点的direction，忽略不符合要求的节点
#     directions_and_ids = [(data['direction'], node) for node, data in G.nodes(data=True)
#                           if 'direction' in data and isinstance(data['direction'], list) and len(data['direction']) == 3]
#     node_directions = np.array([item[0] for item in directions_and_ids])
#     node_ids = np.array([item[1] for item in directions_and_ids])
#     # print(node_directions)
#     # 检查 node_directions 是否为空
#     if node_directions.size == 0:
#         print("Empty!")
#         return G
#
#     # 使用 cKDTree 进行快速范围查询
#     tree = cKDTree(node_directions)
#     indices = tree.query_ball_point(center, radius)
#
#     # 获取要删除的节点
#     nodes_to_remove = node_ids[indices]
#     G.remove_nodes_from(nodes_to_remove)
#     return G
def remove_edges_sphere_nodes_fast(G, center, radius):
    # 提取所有节点的'direction'属性和节点ID
    directions_and_ids = [(data['direction'], node) for node, data in G.nodes(data=True)
                          if 'direction' in data and isinstance(data['direction'], list) and len(data['direction']) == 3]
    node_directions = np.array([item[0] for item in directions_and_ids])
    node_ids = [item[1] for item in directions_and_ids]

    # 检查 node_directions 是否为空
    if node_directions.size == 0:
        return G

    # 使用 cKDTree 进行范围查询
    tree = cKDTree(node_directions)
    indices = tree.query_ball_point(center, radius)

    # 获取要删除的节点ID
    nodes_to_modify = [node_ids[i] for i in indices]

    # 找到与这些节点相连的边并删除它们
    edges_to_remove = [(u, v, k) for u, v, k in G.edges(nodes_to_modify, keys=True)]
    G.remove_edges_from(edges_to_remove)
    return G


def remove_edges_sphere_nodes_fast(G, center, radius):
    # 提取所有节点的'direction'属性和节点ID
    directions_and_ids = [(data['direction'], node) for node, data in G.nodes(data=True)
                          if
                          'direction' in data and isinstance(data['direction'], list) and len(data['direction']) == 3]
    node_directions = np.array([item[0] for item in directions_and_ids])
    node_ids = [item[1] for item in directions_and_ids]

    # 检查 node_directions 是否为空
    if node_directions.size == 0:
        return G

        # 使用 cKDTree 进行范围查询
    tree = cKDTree(node_directions)
    indices = tree.query_ball_point(center, radius)

    # 获取要删除的节点ID
    nodes_to_modify = [node_ids[i] for i in indices]

    # 找到与这些节点相连的边
    edges_to_remove = [(u, v, k) for u, v, k in G.edges(nodes_to_modify, keys=True)]

    # 统计要删除的边数量以及影响的节点数
    num_edges_to_remove = len(edges_to_remove)
    num_nodes_affected = len(nodes_to_modify)

    print(f"Number of edges to remove: {num_edges_to_remove}")
    print(f"Number of nodes affected: {num_nodes_affected}")

    # 删除边
    G.remove_edges_from(edges_to_remove)

    return G
if __name__ == '__main__':
    threshold = 0.95
    # Define LIF model parameters for Drosophila (fruit fly) neurons
    tau_m = 15.0  # 膜时间常数 (ms)
    R_m = 0.5  # 膜电阻 (GΩ)
    V_th = -45.0  # 峰值电位 (mV)
    V_reset = -65.0  # 重置电位 (mV)
    dt = 0.55 # 时间步长 (ms)
    # threshold = 0.8
    center = [270000, 200000, 140000]
    # center = [630000, 200000, 140000]
    radius = 45000
    # 901024 #87624 right - left
    # 438968 #51424
    # 279120 #640
    G = nx.MultiDiGraph()
    print("Load node attributes")
    pos_pair = load_position()
    # print(pos_pair[27833])
    # print(pos_pair)
    for i in range(len(pos_pair)):
        # print(i)
        # print(type(pos_pair[i][3]))
        # print(pos_pair[i][1:4])
        # print(i)
        G.add_node(pos_pair[i][0], direction=np.array(pos_pair[i][1:4], dtype=int).tolist(), flow='', super_class='',
                   _class='', _side='', state=0, V_m=-65)

    print("Load node classifications")
    class_pir = load_classification()
    for i in range(len(class_pir)):
        if class_pir[i][0] in G.nodes.keys():
            G.nodes[class_pir[i][0]]['flow'] = class_pir[i][1]
            G.nodes[class_pir[i][0]]['super_class'] = class_pir[i][2]
            G.nodes[class_pir[i][0]]['_class'] = class_pir[i][3]
            G.nodes[class_pir[i][0]]['_side'] = class_pir[i][8]
    # print(nx.get_node_attributes(G, 'super_class'))
    # print(G.nodes['720575940621762625']['flow'])
    # print(G.nodes['720575940621762625']['super_class'])
    # print(G.nodes['720575940621762625']['_class'])

            # print(111)
    print("Load Edge Attributes")
    node_pair = load_synapse()
    root_id = ''
    next_id = ''
    # for i in range(len(node_pair)):  # node_pair
    #     if node_pair[i][0] != '':
    #         root_id = node_pair[i][0]
    #     else:
    #         pass
    #     if node_pair[i][1] != '':
    #         next_id = node_pair[i][1]
    #     else:
    #         pass
    #     G.add_edge(root_id, next_id, direction=[int(node_pair[i][2]), int(node_pair[i][3]), int(node_pair[i][4])],
    #                state=0, weight=1)
        # G.add_edge(next_id, root_id, direction=[int(node_pair[i][2]), int(node_pair[i][3]), int(node_pair[i][4])],
        #            state=0, weight=1)
    # G = preprocess_graph(G)
    # print(G)
    G = remove_edges_sphere_nodes_fast(G, center, radius)
    print(len(G.nodes))
    print("Origional Graph Done!")
    json_data = nx.node_link_data(nx.create_empty_copy(G, with_data=True))
    filename = "./results_sphere/" + str(32) + ".json"
    with open(filename, 'w') as f:
        json.dump(json_data, f)
    print("Set Initial Values")
    H = nx.DiGraph()
    # total_weight = []
    for node, data in G.nodes(data=True):
        H.add_node(node, **data)
    # 遍历MultiDiGraph，将边的权重相加并添加到简单图中
    for u, v, data in G.edges(data=True):
        if H.has_edge(u, v):
            H[u][v]['weight'] += data['weight']
            # total_weight += data['weight']
        else:
            H.add_edge(u, v, weight=data['weight'])
            # total_weight += data['weight']
        H[u][v]['state'] = 0
    G = H
    for node in G.nodes():

        if G.nodes[node]['flow'] == 'afferent' and G.nodes[node]['super_class'] == 'sensory' and G.nodes[node][
            '_class'] == 'visual' and G.nodes[node]['_side']=='left':
            G.nodes[node]['state'] = 1
    json_data = nx.node_link_data(nx.create_empty_copy(G, with_data=True))
    filename = "./results_sphere/" + str(0) + ".json"
    with open(filename, 'w') as f:
        json.dump(json_data, f)
    print("Main Loop")
    flag = 0
    # RRlist = read_json("RRlist.json")
    while (flag <= 30):
        flag += 1
        nodenum = 0
        for node in G.nodes():
            I_syn = 0
            suum = 0
            sumweight = 0

            temp2 = list(G.out_edges(node))
            for i in range(len(temp2)):
                if abs(G.nodes[node]['state'] - 1) <= 0.0001:
                    G[temp2[i][0]][temp2[i][1]]['state'] = 1
                    # print("Set21")
                else:
                    G[temp2[i][0]][temp2[i][1]]['state'] = 0

            temp = list(G.in_edges(node))
            maxweight = -1
            for i in range(len(temp)):
                # print(G[temp[i][0]][temp[i][1]])
                # print(G[temp[i][0]][temp[i][1]])
                suum += G[temp[i][0]][temp[i][1]]['weight'] * G[temp[i][0]][temp[i][1]]['state']
                sumweight += G[temp[i][0]][temp[i][1]]['weight']
                if G[temp[i][0]][temp[i][1]]['weight'] > maxweight:
                    maxweight = G[temp[i][0]][temp[i][1]]['weight']
                else:
                    pass
                # t2 = G.nodes[node]
                # for j in range(len(t1)):
                # print(t1[j]['state'],end='')
                # p.append(t1[j]['state'])
                # print(1111)
            # print(p)
            if len(temp) == 0:
                # print(p)
                # print("not in the suum")
                G.nodes[node]['state'] = 0
            else:
                # print("value2threshold: %.8f" % (suum/sumweight))
                if suum / maxweight >= threshold:
                    G.nodes[node]['state'] = 1
                else:
                    G.nodes[node]['state'] = 0

            # print("nodenum:%d " % nodenum, end='')
            # print('time cost : %.5f sec' % running_time)
            # reset the initial values
            if G.nodes[node]['flow'] == 'afferent' and G.nodes[node]['super_class'] == 'sensory' and G.nodes[node][
                '_class'] == 'visual' and G.nodes[node]['_side'] =='left':
                G.nodes[node]['state'] = 1
                G.nodes[node]['V_m'] = V_th
                # nodenum += 1
        print("Saving...")
        json_data = nx.node_link_data(nx.create_empty_copy(G, with_data=True))
        filename = "./results_sphere/" + str(flag) + ".json"
        with open(filename, 'w') as f:
            json.dump(json_data, f)


