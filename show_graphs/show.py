import re
import networkx as nx
import matplotlib.pyplot as plt

name_research = 'NCI Anti-cancer activity prediction data'
data_dir = 'NCI'
data_type = 'sdf'

data_path = '{}/data.{}'.format(data_dir, data_type)
title_size = {'fontsize': 40}
count_images = 10
coeff = 10

for ind in range(count_images):
    rows = cols = 2
    fig = plt.figure(figsize=(rows * coeff, cols * coeff))
    list_graphs = []

    for i in range(rows):
        list_graphs.append([])
        for j in range(cols):
            list_graphs[i].append(nx.Graph())

    with open(data_path, 'r') as reader:
        for i in range(rows):
            for j in range(cols):
                ############################################################
                if data_type is 'sdf':
                    pattern = r'\s*(\d*)\s*(\d*)\s*(\d*\s*)*V\d*'
                    pattern_edge = r'\s*(\d*)\s*(\d*)(\s*\d*)*'
                    for line in reader:
                        result = re.search(pattern, line)
                        if result is not None:
                            for k in range(int(result.group(1))):
                                reader.readline()
                            for k in range(int(result.group(2))):
                                result_edge = re.search(pattern_edge, reader.readline())
                                list_graphs[i][j].add_edge(result_edge.group(1), result_edge.group(2))
                            break
                ############################################################
                ############################################################
        plt.title(name_research, title_size)
        plt.axis('off')
        for i in range(rows):
            for j in range(cols):
                fig.add_subplot(rows, cols, i * rows + j + 1)
                nx.draw(list_graphs[i][j])
        plt.savefig('{}/result/{}.png'.format(data_dir, ind + 1))
