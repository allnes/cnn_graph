import re
import networkx as nx
import matplotlib.pyplot as plt

############################################################
# name_research = 'NCI Anti-cancer activity prediction data'
# data_dir = 'NCI'
# data_type = 'sdf'
############################################################
# name_research = 'Twitter Sentiment Graph Data'
# data_dir = 'Twitter'
# data_type = 'nel'
###########################################################
# name_research = 'DBLP Graph Datasets'
# data_dir = 'DBLP'
# data_type = 'nel'
###########################################################
name_research = 'Functional Brain Network Analysis Data'
data_dir = 'fMRI'
data_type = 'nel'
###########################################################

data_path = '{}/data.{}'.format(data_dir, data_type)
title_size = {'fontsize': 40}
count_images = 10
coeff = 10
with open(data_path, 'r') as reader:
    for ind in range(count_images):
        rows = cols = 2
        fig = plt.figure(figsize=(rows * coeff, cols * coeff))
        list_graphs = []
        ############################################################
        if data_type is 'sdf':
            pattern_vertices = r'\s*(\d*)\s*(\d*)\s*(\d*\s*)*V\d*'
            pattern_edge = r'\s*(\d*)\s*(\d*)(\s*\d*)*'
            for line in reader:
                result = re.search(pattern_vertices, line)
                if result is not None:
                    for k in range(int(result.group(1))):
                        reader.readline()
                    gx = nx.Graph()
                    for k in range(int(result.group(2))):
                        result_edge = re.search(pattern_edge, reader.readline())
                        gx.add_edge(result_edge.group(1), result_edge.group(2))
                    list_graphs.append(gx)
                    if len(list_graphs) == rows * cols:
                        break
        ############################################################
        if data_type is 'nel':
            pattern_vertices = r'n\s*(\d*)\s(\w*)'
            pattern_edge = r''
            pattern_end = []
            if data_dir is 'Twitter':
                pattern_edge = r'e\s*(\d*)\s*(\d*)\s*(\d*\.\d*)'
                pattern_end = [r'g\s*\d*\d\s*\d*', r'x\s*\d*']
                N_v = 9
                N_e = 0
            if data_dir is 'DBLP':
                pattern_edge = r'e\s*(\d*)\s*(\d*)\s*(\w*)'
                pattern_end = [r'g\s*\w*\s*\d*', r'x\s*\d*.\d*']
                N_v = 15
                N_e = 15
            if data_dir is 'fMRI':
                pattern_edge = r'e\s*(\d*)\s*(\d*)\s*(\d*)'
                pattern_end = [r'g\s*\w*\s*\d*', r'x\s*\d*.\d*']
                N_v = 15
                N_e = 15
            vertices_name = []
            edge_name = []
            for line in reader:
                result_vert = re.search(pattern_vertices, line)
                result_edge = re.search(pattern_edge, line)
                result_end = re.search(pattern_end[0], line)
                if result_vert is not None:
                    vertices_name.append(result_vert.group(2))
                if result_edge is not None:
                    edge_name.append([result_edge.group(1), result_edge.group(2)])
                if result_end is not None:
                    if len(vertices_name) > N_v and len(edge_name) > N_e:
                        gx = nx.Graph()
                        for edge in edge_name:
                            gx.add_edge(edge[0], edge[1])
                        gx.remove_nodes_from(nx.isolates(gx))
                        list_graphs.append(gx)
                    vertices_name.clear()
                    edge_name.clear()
                if len(list_graphs) == rows * cols:
                    break
        ############################################################
        plt.title(name_research, title_size)
        plt.axis('off')

        for i in range(rows * cols):
                fig.add_subplot(rows, cols, i + 1)
                nx.draw(list_graphs[i])
        plt.savefig('{}/result/{}.png'.format(data_dir, ind + 1))
