# 这是一个示例 Python 脚本。
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import scipy.stats as stats
import random
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
if __name__ == '__main__':
    threshold = 0.8
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
                   _class='', _side='',state = 0)

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
                   state=0, weight = 1)
        # G.add_edge(next_id, root_id, direction=[int(node_pair[i][2]), int(node_pair[i][3]), int(node_pair[i][4])],
        #            state=0, weight=1)
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
    # print(G)
    print("Set Initial Values")

    for node in G.nodes():

        if G.nodes[node]['flow'] == 'afferent' and G.nodes[node]['super_class'] == 'sensory' and G.nodes[node]['_class'] == 'visual' :#and G.nodes[node]['_side']=='right':

            G.nodes[node]['state'] = 1
    print(len(G.nodes))
    print(len(G.edges))
    json_data = nx.node_link_data(nx.create_empty_copy(G, with_data=True))
    filename = "./results_non/" + str(0) + ".json"
    with open(filename, 'w') as f:
        json.dump(json_data, f)
        # print(G.nodes(data=True))
    print("Origional Graph Done!")
        # print(list(G.neighbors('720575940623051957')))#visit the next nodes
        # print(list(G.predecessors('720575940623051957')))
        # print(len(G.out_edges('720575940623051957')))
        # G['720575940630786105']['720575940610739662'][0]['state'] = 1
        # print(G['720575940630786105']['720575940610739662'][0]['state'])
    print("Main Loop")
    flag = 0
        # Begin transform
        '''
        RRlist = []
        while(flag <= 100):
            flag += 1
            nodenum = 0
            for node in G.nodes():
                nodenum += 1
                suum = 0
                temp = list(G.in_edges(node))
                #print(temp)
                #print(temp[0])
                Rlist = []
                Rall = 0
                p = []
                # print(1111)
                for i in range(len(temp)):
                    # print(G[temp[i][0]][temp[i][1]])
                    t1 = G[temp[i][0]][temp[i][1]]
                    t2 = G.nodes[node]
                    for j in range(len(t1)):
                        # rr = np.sqrt(
                        #     (int(t1[j]['x'])-int(t2['nx']))**2 +
                        #     (int(t1[j]['y'])-int(t2['ny']))**2 +
                        #     (int(t1[j]['z'])-int(t2['nz']))**2
                        # )
    
                        rr = np.linalg.norm(np.array(t1[j]['direction']) - np.array(t2['direction']))
    
                        Rall += rr
                        Rlist.append(rr)
                        p.append(t1[j]['state'])
                start = time.time()
                RRlist.append(Rlist)
                for i in range(len(Rlist)):
                    suum += (1-Rlist[i])/Rall*p[i]
                if sum(p) == 0:
                    G.nodes[node]['state'] = 0
                else:
                    print(suum/sum(p))
                    if suum/sum(p) >= threshold:
                        G.nodes[node]['state'] = 1
                    else:
                        G.nodes[node]['state'] = 0
                temp2 = list(G.out_edges(node))
                for i in range(len(temp2)):
                    for j in range(len(G[temp2[i][0]][temp2[i][1]])):
                        G[temp2[i][0]][temp2[i][1]][j]['state'] = G.nodes[node]['state']
    
                end = time.time()
                running_time = end - start
                print("nodenum:%d " % nodenum, end='')
                #print('time cost : %.5f sec' % running_time)
                # reset the initial values
                if G.nodes[node]['flow'] == 'afferent' and G.nodes[node]['super_class'] == 'sensory' and G.nodes[node]['_class'] == 'visual':
                    G.nodes[node]['state'] = 1
            '''
        #RRlist = read_json("RRlist.json")
    while (flag <= 20):
        flag += 1
        nodenum = 0
        for node in G.nodes():
            suum = 0

                # print("nodenum: %d" %nodenum)
                #p = []
                #Rlist = RRlist[nodenum]
                #Rall = sum(Rlist)
            sumweight = 0

            temp2 = list(G.out_edges(node))
            for i in range(len(temp2)):
                    # print(G.nodes[node])
                if abs(G.nodes[node]['state'] - 1) <= 0.001:
                        #print(1111)
                    G[temp2[i][0]][temp2[i][1]]['state'] = 1
                        # print("Set21")
                    else:
                        G[temp2[i][0]][temp2[i][1]]['state'] = 0

                temp = list(G.in_edges(node))
                maxweight = -1
                for i in range(len(temp)):
                    # print(G[temp[i][0]][temp[i][1]])
                    #print(G[temp[i][0]][temp[i][1]])
                    suum += G[temp[i][0]][temp[i][1]]['weight']*G[temp[i][0]][temp[i][1]]['state']
                    sumweight += G[temp[i][0]][temp[i][1]]['weight']
                    if G[temp[i][0]][temp[i][1]]['weight'] > maxweight:
                        maxweight = G[temp[i][0]][temp[i][1]]['weight']
                    else:
                        pass
                    # t2 = G.nodes[node]
                    #for j in range(len(t1)):
                        # print(t1[j]['state'],end='')
                        #p.append(t1[j]['state'])
                        # print(1111)
                #print(p)
                if len(temp) == 0:
                    # print(p)
                    # print("not in the suum")
                    G.nodes[node]['state'] = 0
                else:
                    #print("value2threshold: %.8f" % (suum/sumweight))
                    if suum / maxweight >= threshold:
                        G.nodes[node]['state'] = 1
                    else:
                        G.nodes[node]['state'] = 0

                # print("nodenum:%d " % nodenum, end='')
                # print('time cost : %.5f sec' % running_time)
                # reset the initial values
                if G.nodes[node]['flow'] == 'afferent' and G.nodes[node]['super_class'] == 'sensory' and G.nodes[node]['_class'] == 'visual' :#and G.nodes[node]['_side']=='right':
                    G.nodes[node]['state'] = 1
                # nodenum += 1

            # Save data
            print("Saving...")
            json_data = nx.node_link_data(nx.create_empty_copy(G, with_data=True))
            filename = "./results_v/" + str(flag) + ".json"
            with open(filename, 'w') as f:
                json.dump(json_data, f)

            # np.save('RRlist.npy', RRlist)
            # with open("./RRlist.json", 'w') as fp:
            # json.dump(RRlist, fp)
        # json_data = nx.node_link_data(G)
        # with open('Gmillion.json', 'w') as f:
        # json.dump(json_data, f)
        # nx.write_gml(G,'Gmillion.gml')
        # load_data123
        # array = load_data()
        #
        # G = nx.DiGraph()
        # for i  in range(len(array)):
        #     G.add_nodes(i,poi = [array[i][2],array[i][3],array[i][4]])
        # # G.add_edges_from(array)
        # # G.add_nodes_from(array[:][0])
        # print(G.nodes)
        # #process_data
        # while true

    # 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
