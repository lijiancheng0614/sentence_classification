import re
import torch

from model import Classifier


def evaluate(checkpoint_path,
             num_class,
             num_words,
             datafolds,
             glove=None,
             use_gpu=False):
    checkpoint = torch.load(checkpoint_path)
    config_string = checkpoint['config_string']
    groups = re.search(r'input(\d+)_hidden(\d+)', config_string)
    input_size, hidden_size = int(groups.group(1)), int(groups.group(2))
    classifier = Classifier(input_size, hidden_size, num_class, num_words,
                            glove, use_gpu)
    if use_gpu:
        classifier = classifier.cuda()
    dataset_eval = datafolds[-1]
    classifier.load_state_dict(checkpoint['model'])
    correct, total = classifier.evalute_dataset(dataset_eval)
    return correct, total
