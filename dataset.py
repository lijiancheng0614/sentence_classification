import json
import os
import torch

from tree import TreeNode


def load_glove(pth_path, glove_path, word2id):
    if os.path.exists(pth_path):
        return torch.load(pth_path)
    with open(glove_path) as fd:
        line = fd.readline().rstrip('\n').split(' ')
        dim = len(line[1:])
    glove = torch.zeros(len(word2id), dim)
    words_in_glove = set()
    with open(glove_path) as fd:
        count = 0
        for line in fd:
            line = line.split()
            if line[0] in word2id:
                idx = word2id[line[0]]
                glove[idx] = torch.Tensor(list(map(float, line[1:])))
                words_in_glove.add(line[0])
                count += 1
    print('[INFO] {}/{} words in glove'.format(count, len(word2id)))
    for word, idx in word2id.items():
        if word not in words_in_glove:
            glove[idx].normal_(-0.05, 0.05)
    torch.save(glove, pth_path)
    return glove


def get_json(json_path):
    with open(json_path) as fd:
        return json.load(fd)


def get_labels(labels_path):
    with open(labels_path) as fd:
        return [line.strip().upper() for line in fd]


def build_dataset(json_sentences, labels, dataset, label_map, count_word):
    for sentence, label in zip(json_sentences, labels):
        for token in sentence['tokens']:
            count_word[token['word']] = count_word.get(token['word'], 0) + 1
        if label not in label_map:
            label_map[label] = len(label_map.keys())
        label_id = label_map[label]
        dataset.append((sentence['parse'], label_id))


def process_data_trec(json_paths, labels_paths):
    dataset = list()
    label_map = dict()
    count_word = dict()
    for json_path, labels_path in zip(json_paths, labels_paths):
        json_sentences = get_json(json_path)['sentences']
        labels = get_labels(labels_path)
        build_dataset(json_sentences, labels, dataset, label_map, count_word)
    print('[INFO] {} words, label_map: {}'.format(len(count_word), label_map))
    special_tokens = [u'<BOS>', u'<EOS>', u'<UNK>', u'<PAD>']
    for token in special_tokens:
        count_word[token] = count_word.get(token, 1000)
    id2word = count_word.keys()
    word2id = {w: i for i, w in enumerate(id2word)}
    dataset = [(TreeNode.build(sentence, word2id), label_id)
               for sentence, label_id in dataset]
    for tree_node, label_id in dataset:
        tree_node.compact()
    return dataset, word2id


def process_data_sick(a_json_paths, b_json_paths, labels_paths):

    def __build_dataset(seq_a, seq_b, labels, dataset, count_word):
        for a, b, label in zip(seq_a, seq_b, labels):
            for token in a['tokens'] + b['tokens']:
                count_word[token['word']] = count_word.get(token['word'], 0) + 1
            dataset.append((a['parse'], b['parse'], label))


    def __get_labels(labels_path):
        with open(labels_path, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
        return labels

    dataset = list()
    label_map = dict()
    count_word = dict()

    for a_json_path, b_json_path, labels_path in zip(a_json_paths, b_json_paths, labels_paths):
        a_json_sentences = get_json(a_json_path)['sentences']
        b_json_sentences = get_json(b_json_path)['sentences']
        labels = __get_labels(labels_path)
        datas = list()
        __build_dataset(a_json_sentences, b_json_sentences, labels, datas, count_word)
        dataset.append(datas)

    print('[INFO] {} words, label_map: {}'.format(len(count_word), label_map))
    special_tokens = [u'<BOS>', u'<EOS>', u'<UNK>', u'<PAD>']
    for token in special_tokens:
        count_word[token] = count_word.get(token, 1000)
    id2word = count_word.keys()
    word2id = {w: i for i, w in enumerate(id2word)}

    dataset = [[(TreeNode.build(seq_a, word2id), TreeNode.build(seq_b, word2id), label)
               for seq_a, seq_b, label in datas] for datas in dataset]

    for datas in dataset:
        for node_a, node_b, label_id in datas:
            node_a.compact()
            node_b.compact()
    return dataset, word2id