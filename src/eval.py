import argparse
import sys
from os.path import join

import numpy as np
import torch
from torch.autograd import Variable

from mutils import write_to_csv
from data import get_batch, build_vocab, get_nli


def get_args():
    parser = argparse.ArgumentParser(description='Training NLI model based on just hypothesis sentence')

    # paths
    parser.add_argument("--outputfile", type=str, default="results.csv", help="writes the results on this file.")
    parser.add_argument("--word_emb_dim", type=int, default=300)
    parser.add_argument("--embdfile", type=str, default='../data/embds/glove.840B.300d.txt',
                        help="File containin the word embeddings")
    parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
    parser.add_argument("--model", type=str, help="Input model that has already been trained")
    parser.add_argument("--pred_file", type=str, default='preds', help="Suffix for the prediction files")
    parser.add_argument("--test_path", type=str, required=True, help="The path of the test dataset.")
    parser.add_argument("--train_path", type=str, required=True, help="The path of the train dataset the model is trained on.")
    parser.add_argument("--test_data", type=str, required=True, help="The name of the test dataset", choices=['SNLI',
                       'SNLIHard', 'MNLIMatched', 'MNLIMismatched', 'JOCI', 'SICK-E', 'AddOneRTE', 'DPR',\
                       'FNPLUS', 'SciTail','SPR','MPE', 'QQP', 'GLUEDiagnostic'])
    parser.add_argument("--train_data", type=str, required=True, help="The name of the train dataset",  choices=['SNLI',
                       'SNLIHard', 'MNLIMatched', 'MNLIMismatched', 'JOCI', 'SICK-E', 'AddOneRTE', 'DPR',\
                       'FNPLUS', 'SciTail','SPR','MPE', 'QQP', 'GLUEDiagnostic'])

    # data
    parser.add_argument("--max_train_sents", type=int, default=10000000, help="Maximum number of training examples")
    parser.add_argument("--max_val_sents", type=int, default=10000000, help="Maximum number of validation/dev examples")
    parser.add_argument("--max_test_sents", type=int, default=10000000, help="Maximum number of test examples")
    parser.add_argument("--lorelei_embds", type=int, default=0,
                        help="Whether to use multilingual embeddings released for LORELEI. This requires cleaning up words since wordsare are prefixed with the language. 0 for no, 1 for yes (Default is 0)")

    # model
    parser.add_argument("--batch_size", type=int, default=64)

    # gpu
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")

    # misc
    parser.add_argument("--verbose", type=int, default=1, help="Verbose output")

    params, _ = parser.parse_known_args()

    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(params)

    return params


def compute_score_with_logits(logits, labels, n_classes):
    pred = logits.data.max(1)[1]
    if n_classes == 2:
        pred[pred == 2] = 1
    correct = pred.long().eq(labels.data.long()).cpu().sum()
    return correct


def get_nli_split(path, n_classes, split="test"):
    data = {}
    if n_classes == 3:
        dico_labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2, 'hidden': 3}
    else:
        dico_labels = {'entailment': 0, 'neutral': 1, 'contradiction': 1, 'hidden': 3}
    data['s1'] = [line.rstrip() for line in open(join(path, 's1.' + split), 'r')]
    data['s2'] = [line.rstrip() for line in open(join(path, 's2.' + split), 'r')]
    data['labels'] = np.array([dico_labels[line.rstrip('\n')] for line in open(join(path, 'labels.' + split), 'r')])
    return data


def evaluate(args, nli_net, test_nlipath, n_classes, word_vec, split="test"):
    test = get_nli_split(test_nlipath, n_classes, split)
    for split in ['s1', 's2']:
        test[split] = np.array([['<s>'] +
                                [word for word in sent.split() if word in word_vec] +
                                ['</s>'] for sent in test[split]])

    # Evaluates on the test set.
    correct = 0.
    s1 = test['s1']
    s2 = test['s2']
    target = test['labels']
    outputs = []
    for i in range(0, len(s1), args.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + args.batch_size], word_vec, args.word_emb_dim)

        s2_batch, s2_len = get_batch(s2[i:i + args.batch_size], word_vec, args.word_emb_dim)

        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + args.batch_size])).cuda()
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        outputs.extend(output.data.max(1)[1].cpu().numpy())
        correct += compute_score_with_logits(output, tgt_batch, n_classes)

    eval_acc = round(100 * correct.item() / len(s1), 2)
    print('evaluation accuracy is {0}'.format(eval_acc))
    return eval_acc, outputs


def get_vocab(args):
    # build a vocabulary from all train,dev,test set of the actual snli plus the test set of the
    # all the transfer tasks.
    train, valid, test = {}, {}, {}
    for split in ['test', 'valid', 'train']:
        for s in ['s1', 's2']:
            eval(split)[s] = []
    for datapath, n_classes in [(args.test_path, args.data_to_n_classes[args.test_data]),
                               (args.train_path, args.data_to_n_classes[args.train_data])]:
        transfer_train, transfer_valid, transfer_test = get_nli(datapath, n_classes)
        for split in ['test', 'valid', 'train']:
            for s in ['s1', 's2']:
                eval(split)[s].extend(eval("transfer_" + split)[s])

    word_vec = build_vocab(train['s1'] + train['s2'] +
                           valid['s1'] + valid['s2'] +
                           test['s1'] + test['s2'], args.embdfile)
    return word_vec


def main(args):
    # sets seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.data_to_n_classes = {
        'SNLI': 3, 'SNLIHard': 3, 'MNLIMatched': 3, 'MNLIMismatched': 3, 'JOCI': 3,\
        'SICK-E': 3, 'AddOneRTE': 2, 'DPR': 2, 'FNPLUS': 2, 'SciTail': 2, 'SPR': 2,\
        'MPE': 3, 'QQP': 2,'GLUEDiagnostic': 3
    }

    # builds vocabulary from the all datasets.
    word_vec = get_vocab(args)
    shared_nli_net = torch.load(args.model).eval().cuda()

    eval_accs = {}
    eval_accs["test"] = evaluate(args, shared_nli_net, args.test_path, \
                                 args.data_to_n_classes[args.test_data], word_vec, split="test")[0]
    eval_accs["dev"] = evaluate(args, shared_nli_net, args.test_path, \
                                args.data_to_n_classes[args.test_data], word_vec, split="dev")[0]
    write_to_csv(eval_accs, args, args.outputfile)


if __name__ == '__main__':
    args = get_args()
    main(args)
