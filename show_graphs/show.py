import re
import networkx as nx
import matplotlib.pyplot as plt

data_path = 'NCI/data.sdf'
data_type = 'sdf'
total_count = 10

with open(data_path, 'r') as reader:
    for i in range(total_count):
        gr = nx.Graph()
        ############################################################
        if data_type is 'sdf':
            pattern = r'\s*(\d*)\s*(\d*)\s*(\d*\s*)*V\d*'
            pattern_edge = r'\s*(\d*)\s*(\d*)(\s*\d*)*'
            for line in reader:
                result = re.search(pattern, line)
                if result is not None:
                    for i in range(int(result.group(1))):
                        reader.readline()
                    for i in range(int(result.group(2))):
                        result_edge = re.search(pattern_edge, reader.readline())
                        gr.add_edge(result_edge.group(1), result_edge.group(2))
                    break
        ############################################################
        ############################################################
        nx.draw(gr)
        plt.show()
