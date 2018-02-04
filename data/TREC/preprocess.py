RAW_PATHS = ['train_5500.label', 'TREC_10.label']
SENTENCES_PATHS = ['sentences_train.txt', 'sentences_test.txt']
LABELS_PATHS = ['labels_train.txt', 'labels_test.txt']


def load_data(path):
    fd = open(path, 'rb')
    _lines = [line.strip() for line in fd]
    fd.close()
    return _lines


def get_sentences_and_labels(text):
    _labels = [t.split()[0].split(':')[0] for t in text]
    _sentences = [t.split()[1:] for t in text]
    _sentences = [' '.join(t) for t in _sentences]
    return _sentences, _labels


def save(path, lines):
    fd = open(path, 'w')
    for line in lines:
        fd.write(line + '\n')
    fd.close()


for raw_path, sentences_path, labels_path in zip(RAW_PATHS, SENTENCES_PATHS,
                                                 LABELS_PATHS):
    print(raw_path, sentences_path, labels_path)
    data = load_data(raw_path)
    sentences, labels = get_sentences_and_labels(data)
    save(sentences_path, sentences)
    save(labels_path, labels)
