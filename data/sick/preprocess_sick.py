"""
Preprocessing script for SICK data.
"""

import os
import glob


def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')


def split(filepath, dst_dir):
    with open(filepath) as datafile, \
            open(os.path.join(dst_dir, 'a.txt'), 'w') as afile, \
            open(os.path.join(dst_dir, 'b.txt'), 'w') as bfile,  \
            open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
            open(os.path.join(dst_dir, 'sim.txt'), 'w') as simfile:
        datafile.readline()
        for line in datafile:
            i, a, b, sim, ent = line.strip().split('\t')
            idfile.write(i + '\n')
            afile.write(a + '\n')
            bfile.write(b + '\n')
            simfile.write(sim + '\n')


if __name__ == '__main__':

    print('Preprocessing SICK dataset')

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    lib_dir = os.path.join(base_dir, '..')
    train_dir = os.path.join(data_dir, 'train')
    dev_dir = os.path.join(data_dir, 'dev')
    test_dir = os.path.join(data_dir, 'test')
    make_dirs([train_dir, dev_dir, test_dir])

    split(os.path.join(data_dir, 'SICK_train.txt'), train_dir)
    split(os.path.join(data_dir, 'SICK_trial.txt'), dev_dir)
    split(os.path.join(data_dir, 'SICK_test_annotated.txt'), test_dir)

