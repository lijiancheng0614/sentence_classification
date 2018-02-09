import argparse
import os
import random
import torch

from dataset import load_glove, process_data_trec
from trainer import train
from evaluator import evaluate


def solve(args):
    if args.task == 'TREC':
        num_class = 6
        data_dir = os.path.join(args.data_dir, 'TREC')
        json_paths = [
            os.path.join(data_dir, 'sentences_train.txt.json'),
            os.path.join(data_dir, 'sentences_test.txt.json')
        ]
        labels_paths = [
            os.path.join(data_dir, 'labels_train.txt'),
            os.path.join(data_dir, 'labels_test.txt')
        ]
        dataset, word2id = process_data_trec(json_paths, labels_paths)
        datafolds = [dataset[:-500], dataset[-500:]]
        num_folds = 2
    else:
        print('[ERROR] Unknown task')
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.glove_path == '':
        glove = None
        input_size = 100
        hidden_size = 50
    else:
        glove = load_glove('train_glove_{}.pth'.format(args.task),
                           args.glove_path, word2id)
        input_size = 300
        hidden_size = 100
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)
    try:
        lr_milestones = map(int, args.lr_milestones.split(','))
    except:
        print('[WARN] Cannot parse lr_milestones {}'.format(
            args.lr_milestones))
        lr_milestones = None
    if args.phase == 'train':
        train(
            args.task,
            num_class,
            len(word2id),
            args.logs_dir,
            args.models_dir,
            datafolds,
            num_folds,
            glove,
            args.epochs,
            args.batch_size,
            input_size,
            hidden_size,
            args.lr,
            lr_milestones,
            args.weight_decay,
            use_gpu=args.gpu)
    else:
        correct, total = evaluate(args.checkpoint_path, num_class,
                                  len(word2id), datafolds, glove, args.gpu)
        print('{} / {} = {:.3f}'.format(correct, total,
                                        float(correct) / total))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='TREC', help='Dataset name')
    parser.add_argument(
        '--phase',
        default='train',
        choices=['train', 'eval'],
        help='Phase: train/eval')
    parser.add_argument(
        '--checkpoint_path',
        default='',
        help='Checkpoint path used in evaluation')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--logs_dir', default='logs')
    parser.add_argument('--models_dir', default='models')
    parser.add_argument(
        '--epochs', type=int, default=40, help='Maximum epochs')
    parser.add_argument(
        '--batch_size', type=int, default=25, help='Batch size')
    parser.add_argument(
        '--lr', type=float, default=0.008, help='Initial learning rate')
    parser.add_argument(
        '--lr_milestones', default='11', help='Milestones for MultiStepLR')
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument(
        '--glove_path',
        default='data/glove.840B.300d.txt',
        help='GloVe path, leave empty string if GloVe is not used')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--seed', type=int, default=10137)
    args = parser.parse_args()
    solve(args)


if __name__ == '__main__':
    main()
