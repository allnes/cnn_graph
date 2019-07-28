import numpy as np


def get_twitter_data_set():
    data_nel = open('new_data/TWITTER-Real-Graph-Partial.nel', 'r')
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
                graph = np.zeros((graph_size, graph_size)).astype(np.float32)
                flag_create_matrix = False
            graph[int(split_line[1]) - 1][int(split_line[2]) - 1] = np.float32(split_line[3])

        if flag_nel == 'x':
            data_graph.append(graph)
            labels_graph.append(split_line[1])
            s.add(split_line[1])

    len_max = len(data_graph[0])
    for matr_new in data_graph:
        len_max = max(len_max, len(matr_new))

    new_data_graph = np.ndarray((len(data_graph), 1))
    for matr_2 in data_graph:
        edit_size = len_max - len(matr_2)
        new_data_graph = np.append(new_data_graph, np.pad(matr_2, ((0, edit_size), (0, edit_size)), mode='constant', constant_values=(0, 0)))

    new_labels_graph = np.ndarray((len(data_graph), 1))
    new_labels_graph = np.append(new_labels_graph, labels_graph)

    return {"data": new_data_graph, "labels": new_labels_graph, "label_values": s}


def test_twitter_data_set():
    print("Twitter data:")
    curr_data_twitter = get_twitter_data_set()
    print(len(curr_data_twitter["data"]))
    print(len(curr_data_twitter["labels"]))
    print(curr_data_twitter["label_values"])


def get_dblp_data_set():
    data_nel = open("new_data/DBLP_v1.nel", 'r')
    graph_size = 0
    graph = 0
    flag_create_matrix = True

    data_graph = []
    labels_graph = []
    dict_elem = {'P2P': 1.0, 'P2W': 0.1, 'W2W': -1.0}

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
                graph = np.zeros((graph_size, graph_size)).astype(np.float32)
                flag_create_matrix = False
            graph[int(split_line[1]) - 1][int(split_line[2]) - 1] = np.float32(dict_elem[split_line[3]])

        if flag_nel == 'x':
            data_graph.append(graph)
            labels_graph.append(split_line[1])
            s.add(split_line[1])
    return {"data": data_graph, "labels": labels_graph, "label_values": s}


def test_dblp_data_set():
    print("\nDBLP brain data:")
    curr_data_dblp = get_dblp_data_set()
    print(len(curr_data_dblp["data"]))
    print(len(curr_data_dblp["labels"]))
    print(curr_data_dblp["label_values"])


if __name__ == "__main__":
    dataset = get_twitter_data_set()
    for matr in dataset["data"]:
        assert len(matr) == len(dataset["data"][0])
