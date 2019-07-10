import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--type', type=str,
                    help='nel-file for converting')
type_dataset = parser.parse_args().type


def get_twitter_data_set():
    data_nel = open("new_data/TWITTER-Real-Graph-Partial.nel", 'r')
    graph_size = 0
    graph = 0
    flag_create_matrix = True

    data_graph = []
    labels_graph = []

    s = set()

    for line in data_nel:
        split_line = line.split()

        if len(split_line) == 0:
            graph_size = 0
            flag_create_matrix = True
            continue

        flag_nel = split_line[0]

        if flag_nel == 'n':
            graph_size += 1

        if flag_nel == 'e':
            if flag_create_matrix:
                graph = np.zeros((graph_size, graph_size))
                flag_create_matrix = False
            graph[int(split_line[1]) - 1][int(split_line[2]) - 1] = np.float64(split_line[3])

        if flag_nel == 'x':
            data_graph.append(graph)
            labels_graph.append(split_line[1])
            s.add(split_line[1])
    return data_graph, labels_graph, s


if type_dataset == "twitter":
    curr_data = get_twitter_data_set()
    print(len(curr_data[0]))
    print(len(curr_data[1]))
    print(curr_data[2])
