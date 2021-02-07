import argparse
import sys

import torch.nn.functional
from torch import nn
from torch.utils.data import DataLoader

from datasets import WordIndexDataset, MailDirDatasetEval, IndexVectorCollatorEval
from models import SpamClassifier
from settings import VOCABULARY_SIZE, MAX_MESSAGE_LENGTH_WORDS
from utils import load_classifier


def classify_messages(classifier_state_file: str, message_dir: str):
    model = SpamClassifier(VOCABULARY_SIZE)
    model = model.cuda()
    vocabulary = load_classifier(classifier_state_file, model)

    print('Loaded classifier state from:', classifier_state_file, file=sys.stderr)

    messages = MailDirDatasetEval(message_dir, language='en')
    samples = WordIndexDataset(messages, vocabulary, MAX_MESSAGE_LENGTH_WORDS)
    samples_loader = DataLoader(samples, batch_size=100, collate_fn=IndexVectorCollatorEval(), num_workers=8)

    print('Processing messages in:', message_dir, file=sys.stderr)

    spam, ham = 0, 0

    for x, x_len, path in samples_loader:
        y_pred = model(x.cuda(), x_len).cpu()
        spam_score = nn.functional.softmax(y_pred, dim=-1)[:, 1]
        is_spam = spam_score > 0.5
        spam += is_spam.sum().item()
        ham += (is_spam == False).sum().item()
        for s_s, i_s, p in zip(spam_score, is_spam, path):
            print(f'{p}: {"SPAM" if i_s.item() else "HAM"}, score: {s_s.item()}')
    print(f'Statistics: spam: {spam}, ham: {ham}, total: {spam + ham}', file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify messages.')
    parser.add_argument('-c', '--classifier', type=str, required=True,
                        help='Classifier state file (created by train.py)')
    parser.add_argument('-m', '--message_dir', type=str, required=True,
                        help='Directory with messages to process.')
    args = parser.parse_args()

    classify_messages(args.classifier, args.message_dir)
