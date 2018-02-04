import re


class TreeNode(object):
    def __init__(self):
        self.children = list()
        self.node_type = None
        self.word = None

    def add_child(self, tree):
        self.children.append(tree)

    def is_leaf(self):
        return len(self.children) == 0

    def compact(self):
        if len(self.children) == 1:
            if self.children[0].is_leaf():
                self.node_type = self.children[0].node_type
                self.word = self.children[0].word
                self.children = list()
            else:
                self.children = self.children[0].children
                self.compact()
        elif len(self.children) > 1:
            for child in self.children:
                child.compact()

    def tolist(self):
        if self.is_leaf():
            return [self]
        result = list()
        for child in self.children:
            result += child.tolist()
        return result

    @staticmethod
    def build(parsed_sentence, word2id):
        sentence = re.sub(' +', ' ', parsed_sentence.replace('\n', ''))
        stack = list()
        i = 0
        while i < len(sentence):
            if sentence[i] == '(':
                stack.append(TreeNode())
                i += 1
            elif sentence[i] == ')':
                if len(stack) == 1:
                    return stack[0]
                stack[-2].add_child(stack[-1])
                stack.pop()
                i += 1
            elif sentence[i] == ' ':
                i += 1
            else:
                j = i + 1
                while j < len(sentence) and sentence[j] not in ['(', ')', ' ']:
                    j += 1
                word = sentence[i:j]
                if stack[-1].node_type is None:
                    stack[-1].node_type = word
                else:
                    stack[-1].word = word2id[word]
                i = j
