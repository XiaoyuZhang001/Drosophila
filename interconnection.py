
# 读取数据
import pandas as pd
import networkx as nx
import numpy as np
import json
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
# class Neutron(object):  # 创建Circle类
#    def __init__(self,x,y,z,next = None):
#        self.x = x
#        self.y = y
#        self.z = z
#        self.next = next
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
def load_cell():
    # node_pair1 = np.loadtxt('./coordinates.txt', dtype=bytes)
    class_pair = []
    with open('./cell_stats.txt', "rb") as f:
        for line in f.readlines():  # 打开后逐行读取
            each_line = line.decode().rstrip().split(',')
            # print(each_line)
            # each_line[1] = each_line[1].split(' ')
            class_pair.append(each_line)
    # node_pair = node_pair[1:]
    # print(pos_pair[10])
    # print(len(pos_pair[10]))
    return class_pair

def extract_edge_attribute(G, attribute):
    attributes = []
    for u, v, data in G.edges(data=True):
        if attribute in data:
            attributes.append(data[attribute])
        else:
            attributes.append(None)  # 如果某些边没有这个属性，可以选择添加None或者其他默认值
    return attributes


def mse(matrix1, matrix2):
    return np.mean((matrix1 - matrix2) ** 2)
def jaccard_similarity(matrix1, matrix2):
    similarities = []
    for i in range(min(len(matrix1), len(matrix2))):
        score = jaccard_score(matrix1[i], matrix2[i], average='binary')
        similarities.append(score)
    return np.mean(similarities)
def jaccard_similarity_3d(matrix1, matrix2):
    total_score = 0
    count = 0
    for i in range(min(len(matrix1), len(matrix2))):
        for j in range(min(len(matrix1[i]), len(matrix2[i]))):
            # 使用 'macro' 作为平均方法来处理多类别数据
            score = jaccard_score(matrix1[i][j], matrix2[i][j], average='macro')
            total_score += score
            count += 1
    return total_score / count if count > 0 else 0

def cosine_sim(matrix1, matrix2):
    # 确保两个矩阵的形状相同
    if matrix1.shape[1] != matrix2.shape[1]:
        raise ValueError("Matrices have different dimensions!")
    return mean(cosine_similarity(matrix1, matrix2))


def cosine_similarity_3d(matrix1, matrix2):
    total_sim = 0
    count = 0
    for i in range(min(len(matrix1), len(matrix2))):
        for j in range(min(len(matrix1[i]), len(matrix2[i]))):
            cos_sim = cosine_similarity([matrix1[i][j]], [matrix2[i][j]])[0][0]
            total_sim += cos_sim
            count += 1
    return total_sim / count if count > 0 else 0
def sqrt_normalization(data):
    data_sqrt = np.sqrt(data)  # 取平方根
    min_val = np.min(data_sqrt)
    max_val = np.max(data_sqrt)
    return (data_sqrt - min_val) / (max_val - min_val) /2 # 线性归一化

def mutual_information_3d(array1, array2):
    # 确保数组维度相同
    if array1.shape != array2.shape:
        raise ValueError("Arrays must have the same shape.")

    # 展平每个二维层，以便进行互信息计算
    flattened_layers1 = [layer.flatten() for layer in array1]
    flattened_layers2 = [layer.flatten() for layer in array2]

    # 使用并行计算每层的互信息
    mi_scores = Parallel(n_jobs=-1)(delayed(mutual_info_score)(flattened_layers1[i], flattened_layers2[i]) for i in range(len(flattened_layers1)))

    # 计算平均互信息
    average_mi = np.mean(mi_scores)
    return average_mi


import networkx as nx
import random


def preserve_degree_rewire(graph):
    # 确保图是MultiDiGraph
    if not isinstance(graph, nx.MultiDiGraph):
        raise ValueError("输入的图必须是MultiDiGraph")

    # 创建一个新图，复制原图的节点和节点属性
    new_graph = nx.MultiDiGraph()
    new_graph.add_nodes_from(graph.nodes(data=True))

    # 从原图中收集所有的边（包括属性）
    edges = [(u, v, data) for u, v, data in graph.edges(data=True)]

    # 打乱边的顺序
    random.shuffle(edges)

    # 重新分配边，保持每个节点的度数不变
    out_degree = {node: len(list(graph.out_edges(node))) for node in graph.nodes()}
    in_degree = {node: len(list(graph.in_edges(node))) for node in graph.nodes()}

    available_targets = [node for node in graph.nodes() for _ in range(in_degree[node])]
    random.shuffle(available_targets)

    for u, _, data in edges:
        if out_degree[u] > 0:
            v = available_targets.pop()
            new_graph.add_edge(u, v, **data)
            out_degree[u] -= 1

    return new_graph

if __name__ == '__main__':
    random.seed(1)
    input_type = 'visual'
    # pd.set_option('display.max_columns', None)
    # 显示所有行
    # pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
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
                   _class='', _side='',size = '',area = '',length = '', region = [-1,-1,-1],state = 0)

    print("Load node classifications")
    class_pir = load_classification()
    for i in range(len(class_pir)):
        if class_pir[i][0] in G.nodes.keys():
            G.nodes[class_pir[i][0]]['flow'] = class_pir[i][1]
            G.nodes[class_pir[i][0]]['super_class'] = class_pir[i][2]
            G.nodes[class_pir[i][0]]['_class'] = class_pir[i][3]
            G.nodes[class_pir[i][0]]['_side'] = class_pir[i][8]
    cell_pair = load_cell()
    for i in range(len(cell_pair)):#for i in range(len(cell_pair)):
        if cell_pair[i][0] in G.nodes.keys():
            G.nodes[cell_pair[i][0]]['length'] = cell_pair[i][1]
            G.nodes[cell_pair[i][0]]['area'] = cell_pair[i][2]
            G.nodes[cell_pair[i][0]]['size'] = cell_pair[i][3]

    print("Load Edge Attributes")
    node_pair = load_synapse()
    root_id = ''
    next_id = ''
    for i in range(len(node_pair)):  # node_pair
        if node_pair[i][0] != '':
            root_id = node_pair[i][0]
        else:
            pass
        if node_pair[i][1] != '':
            next_id = node_pair[i][1]
        else:
            pass
        G.add_edge(root_id, next_id, direction=[int(node_pair[i][2]), int(node_pair[i][3]), int(node_pair[i][4])],
                   state=0,region = [-1,-1,-1], weight = 1)
    edges_data = {
        'class_1_edges': [],
        'class_2_edges': [],
        'mixed_class_edges': []
    }
    for u, v, data in G.edges(data=True):
        u_class = G.nodes[u]['_class']
        v_class = G.nodes[v]['_class']

        if u_class == 'visual' and v_class == 'visual':
            # Class 1 edge
            edges_data['class_1_edges'].append({'direction': data['direction'], 'label': 1})
        elif (u_class in {'ALLN', 'ALIN', 'ALON', 'ALPN', 'olfactory'} and
              v_class in {'ALLN', 'ALIN', 'ALON', 'ALPN', 'olfactory'}):
            # Class 2 edge
            edges_data['class_2_edges'].append({ 'direction': data['direction'], 'label': 2})
        else:
            # Mixed class edge (between class 1 and class 2 nodes)
            edges_data['mixed_class_edges'].append({ 'direction': data['direction'], 'label': 3})

            # Save to JSON
    with open('edges_data.json', 'w') as f:
        json.dump(edges_data, f, indent=4)