import os
import pathlib
import pickle
import random
import re
import sys
import unicodedata
from collections import defaultdict
from operator import itemgetter
from typing import Dict

import bs4
import mailparser
import nltk
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


class BadMessage(ValueError):
    pass


def read_email(message_path):
    message = mailparser.parse_from_file(message_path)
    text = ' '.join(message.text_html) if message.text_html else ' '.join(message.text_plain)
    if not text:
        raise BadMessage("invalid content")
    parsed = bs4.BeautifulSoup(text, features='lxml')
    return parsed.get_text()


def normalize(text: str, language):
    words = nltk.word_tokenize(text)
    words = [unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode('utf-8', 'ignore') for w in words]
    words = [w.lower() for w in words]
    words = [re.sub(r'([^\w\s]|\d)', '', w) for w in words]
    words = [w for w in words if w != '' and len(w) < 25]  # remove empty and very long words

    if language == 'en':
        stemmer = nltk.SnowballStemmer(language='english')
        words = [stemmer.stem(w) for w in words]
    if len(words) == 0:
        raise BadMessage('empty after normalize')
    return words


def process_email(message_path, language):
    return normalize(read_email(message_path), language)


class MailDirDataset(Dataset):
    def __init__(self, root_path: str, test=False, shuffle=True, language='en') -> None:
        self.root_path = root_path
        self.language = language
        self.spam_dir = os.path.join(root_path, 'spam')
        self.ham_dir = os.path.join(root_path, 'ham')

        spam_files = list(str(p) for p in pathlib.Path(self.spam_dir).rglob('*') if p.is_file())
        ham_files = list(str(p) for p in pathlib.Path(self.ham_dir).rglob('*') if p.is_file())

        train_spam_files, test_spam_files = train_test_split(spam_files, test_size=0.2)
        train_ham_files, test_ham_files = train_test_split(ham_files, test_size=0.2)

        self.spam_files = test_spam_files if test else train_spam_files
        self.ham_files = test_ham_files if test else train_ham_files

        spam_data = [(m, 1) for m in self.spam_files]
        ham_data = [(m, 0) for m in self.ham_files]
        self.dataset = spam_data + ham_data

        if shuffle:
            random.shuffle(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        try:
            return normalize(read_email(x), language=self.language), y
        except BadMessage as ex:
            print(f'File {x} ignored: {str(ex)}', file=sys.stderr)
            return None, None

    def __len__(self):
        return len(self.dataset)

    def __str__(self) -> str:
        return f'MailDirDataset:{self.root_path}'


def pre_cache_dataset(dataset: Dataset, save_location: str):
    loader = DataLoader(dataset, batch_size=None, num_workers=10)
    data = []
    last_proc = 0
    for idx, (x, y) in enumerate(loader):
        proc = idx * 100 // len(loader)
        if proc > last_proc:
            last_proc = proc
            print('{}%'.format(proc), flush=True)
        if x is not None:
            data.append((x, y))
    with open(save_location, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved cache for {str(dataset)} to {save_location}')


class PreCachedDataset(Dataset):
    def __init__(self, save_file: str) -> None:
        with open(save_file, 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def create_vocabulary(dataset: Dataset, vocabulary_size=2048):
    common_words = defaultdict(int)
    for x, _ in dataset:
        for w in x:
            common_words[w] += 1
    common_words = sorted(common_words.items(), key=itemgetter(1), reverse=True)
    return {word: idx + 2 for idx, (word, _) in enumerate(common_words[:vocabulary_size - 2])}


class WordIndexDataset(Dataset):
    PADDING = 0
    UNKNOWN = 1

    def __init__(self, word_dataset: PreCachedDataset, vocabulary: Dict[str, int], max_words=250):
        self.base_data = word_dataset
        self.vocabulary = vocabulary
        self.max_words = max_words

    def __getitem__(self, index):
        x, y = self.base_data[index]
        x = [self.vocabulary.get(word, self.UNKNOWN) for word in x[:self.max_words]]
        return torch.LongTensor(x), len(x), y

    def __len__(self):
        return len(self.base_data)


class IndexVectorCollator:
    def __call__(self, batch):
        words, lens, targets = zip(*batch)
        words = pad_sequence(words, padding_value=0)
        return words, lens, torch.LongTensor(targets)


if __name__ == '__main__':
    pre_cache_dataset(MailDirDataset('data/enron', language='en', test=False), 'data/enron.train.pck')
    pre_cache_dataset(MailDirDataset('data/enron', language='en', test=True), 'data/enron.test.pck')
    pre_cache_dataset(MailDirDataset('data/spamassassin', language='en', test=False), 'data/spamassassin.train.pck')
    pre_cache_dataset(MailDirDataset('data/spamassassin', language='en', test=True), 'data/spamassassin.test.pck')
