import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TreeNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 vocab_size,
                 glove=None,
                 use_gpu=False):
        super(TreeNet, self).__init__()
        self.use_gpu = use_gpu
        self.hidden_size = hidden_size
        if glove is None:
            self.word_embeddings = nn.Embedding(vocab_size, input_size)
            self.word_embeddings.weight.requires_grad = True
        else:
            glove_vocab_size, glove_input_size = glove.size()
            self.word_embeddings = nn.Embedding(glove_vocab_size,
                                                glove_input_size)
            self.word_embeddings.weight.data.copy_(glove)
            self.word_embeddings.weight.requires_grad = False
        self.encode_x = nn.Linear(input_size, hidden_size * 4)
        self.encode_h = nn.Linear(hidden_size, hidden_size * 4)
        self.encode_s = nn.Linear(hidden_size, hidden_size * 3)
        self.encode_c = nn.Linear(hidden_size, hidden_size * 3)

    def init_state(self):
        hidden = torch.zeros(1, self.hidden_size)
        cell = torch.zeros(1, self.hidden_size)
        if self.use_gpu:
            hidden, cell = hidden.cuda(), cell.cuda()
        return Variable(
            hidden, requires_grad=False), Variable(
                cell, requires_grad=False)

    def node_forward(self, inputs, hidden_x, cell_x):
        gates = self.encode_h(hidden_x) + self.encode_x(inputs)
        i, o, f, c = gates.chunk(4, 1)
        i, o, f, c = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f), F.tanh(c)
        cell = f * cell_x + i * c
        hidden = o * F.tanh(cell)
        return hidden, cell

    def forward(self, tree_node, state=None):
        if state is None:
            state = self.init_state()
        hidden_x, cell_x = state
        if tree_node.is_leaf():
            word = torch.LongTensor([tree_node.word])
            if self.use_gpu:
                word = word.cuda()
            word = Variable(word, requires_grad=False)
            embeds = self.word_embeddings(word)
            hidden_child, cell_child = self.node_forward(
                embeds, *self.init_state())
            gates = self.encode_s(hidden_x) + self.encode_c(hidden_child)
        else:
            self.forward(tree_node.children[0])
            for i in range(1, len(tree_node.children)):
                self.forward(tree_node.children[i],
                             tree_node.children[i - 1].state)
            hidden_child, cell_child = tree_node.children[-1].state
            gates = self.encode_s(hidden_x) + self.encode_c(hidden_child)
        i, o, f = gates.chunk(3, 1)
        i, o, f = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f)
        cell = f * cell_x + i * cell_child
        hidden = o * F.tanh(cell)
        tree_node.state = hidden, cell
        return hidden


class Classifier(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 vocab_size,
                 glove=None,
                 use_gpu=False):
        super(Classifier, self).__init__()
        self.unit = TreeNet(input_size, hidden_size, vocab_size, glove,
                            use_gpu)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, tree_node):
        hidden = self.unit(tree_node)
        prediction = self.classifier(hidden)
        return prediction

    def evalute(self, tree_node, label):
        prediction = self.forward(tree_node)
        _, idx = prediction.cpu().max(1)
        return label == int(idx[0].data[0])

    def evalute_dataset(self, dataset):
        results = [
            self.evalute(tree_root, label) for tree_root, label in dataset
        ]
        correct = sum(results)
        return correct, len(dataset)
