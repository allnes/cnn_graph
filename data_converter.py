import numpy as np


def get_twitter_data_set():
    data_nel = open('new_data/TWITTER-Real-Graph-Partial.nel', 'r')
    len_max = 0
    for line in data_nel:
        split_line = line.split()
        if len(split_line) == 0:
            continue
        flag_nel = split_line[0]
        if flag_nel == 'e':
            len_max = max(len_max, int(split_line[1]), int(split_line[2]))
    data_nel.close()

    data_nel = open('new_data/TWITTER-Real-Graph-Partial.nel', 'r')
    graph = 0
    flag_create_matrix = True

    data_graph = []
    labels_graph = []

    s = set()

    # max_elem = np.finfo(np.float32).min
    # min_elem = np.finfo(np.float32).max
    for line in data_nel:
        split_line = line.split()

        if len(split_line) == 0:
            flag_create_matrix = True
            continue

        flag_nel = split_line[0]

        if flag_nel == 'e':
            if flag_create_matrix:
                graph = np.zeros((len_max, len_max)).astype(np.float32)
                flag_create_matrix = False
            # max_elem = np.maximum(max_elem, np.float32(split_line[3]))
            # min_elem = np.minimum(min_elem, np.float32(split_line[3]))
            wght = np.float32(split_line[3]) * 1000
            if 0.1 < wght < 2.25:
                new_weight = 150
            elif 2.25 <= wght < 4.5:
                new_weight = 55
            elif 4.5 <= wght < 6.75:
                new_weight = 10
            else:
                new_weight = 100
            graph[int(split_line[1]) - 1][int(split_line[2]) - 1] = new_weight

        if flag_nel == 'x':
            data_graph.append(graph)
            labels_graph.append(split_line[1])
            s.add(split_line[1])
    return {"data": data_graph, "labels": labels_graph, "label_values": s}


def test_twitter_data_set():
    print("Twitter data:")
    curr_data_twitter = get_twitter_data_set()
    print(len(curr_data_twitter["data"]))
    print(len(curr_data_twitter["labels"]))
    print(curr_data_twitter["label_values"])


def get_dblp_data_set():
    data_nel = open("new_data/DBLP_v1.nel", 'r')
    graph = 0
    flag_create_matrix = True

    len_max = 0
    for line in data_nel:
        split_line = line.split()
        if len(split_line) == 0:
            continue
        flag_nel = split_line[0]
        if flag_nel == 'e':
            len_max = max(len_max, int(split_line[1]), int(split_line[2]))
    data_nel.close()

    data_nel = open("new_data/DBLP_v1.nel", 'r')
    data_graph = []
    labels_graph = []
    dict_elem = {'P2P': 10, 'P2W': 55, 'W2W': 100}

    s = set()

    for line in data_nel:
        split_line = line.split()

        if len(split_line) == 0:
            flag_create_matrix = True
            continue

        flag_nel = split_line[0]

        if flag_nel == 'e':
            if flag_create_matrix:
                graph = np.zeros((len_max, len_max)).astype(np.float32)
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
    print("start check")
    for matr in dataset["data"]:
        assert len(matr) == len(dataset["data"][0])
    test_twitter_data_set()
    test_dblp_data_set()
