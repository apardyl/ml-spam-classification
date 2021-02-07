import argparse
import copy
import datetime
import os
import shutil
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from datasets import PreCachedDataset, create_vocabulary, WordIndexDataset, IndexVectorCollator
from models import SpamClassifier
from settings import MAX_MESSAGE_LENGTH_WORDS, VOCABULARY_SIZE, DATA_DIRECTORY
from utils import load_train_state, save_train_state, FocalLoss

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 100

LR = 0.005

CHECKPOINT_PREFIX = 'checkpoint'
LOGS_DIR = 'runs'
SAVED_MODELS_PATH = 'saved_models'


def evaluate(model: nn.Module, test_loader: torch.utils.data.DataLoader, loss_fn=None,
             writer: SummaryWriter = None, epoch: int = -1):
    with torch.no_grad():
        model.eval()
        epoch_losses = []
        epoch_accuracy = []
        conf_matrix = np.zeros((2, 2), dtype='int32')
        for step, (x, x_len, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda()
            y_pred = model(x, x_len)
            y_argmax = torch.argmax(y_pred, 1)
            accuracy = y_argmax.eq(y).sum().item() / y.shape[0]
            epoch_accuracy.append(accuracy)
            conf_matrix += confusion_matrix(y.cpu().numpy(), y_argmax.cpu().numpy())
            if loss_fn:
                loss_val = loss_fn(y_pred, y)
                epoch_losses.append(loss_val.item())
            print('    Test batch {} of {}'.format(step + 1, len(test_loader)), file=sys.stderr)

        print(f'Test loss: {np.mean(epoch_losses):.4f}, accuracy: {np.mean(epoch_accuracy):.4f}')
        if writer:
            writer.add_scalar('Loss/test', np.mean(epoch_losses), global_step=epoch)
            writer.add_scalar('Accuracy/test', np.mean(epoch_accuracy), global_step=epoch)
            conf_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['ham', 'spam']).plot(
                cmap='Blues', values_format='d')
            writer.add_figure('ConfusionMatrix', conf_plot.figure_, global_step=epoch, close=True)
        return np.mean(epoch_accuracy)


def train_model(train_data_words, test_data_words, model, epochs=30):
    log_file = os.path.join(LOGS_DIR, f'{model.__class__.__name__}.{str(train_data_words)}')
    checkpoint_file = f'{CHECKPOINT_PREFIX}.{model.__class__.__name__}.{str(train_data_words)}'

    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if os.path.exists(checkpoint_file):
        print('Loading checkpoint')
        epoch, best_score, vocabulary = load_train_state(checkpoint_file, model, optimizer, scheduler)
    else:
        epoch = 0
        best_score = -1
        vocabulary = create_vocabulary(train_data_words, vocabulary_size=VOCABULARY_SIZE)

    best_model = copy.deepcopy(model)

    train_data = WordIndexDataset(train_data_words, vocabulary, max_words=MAX_MESSAGE_LENGTH_WORDS)
    test_data = WordIndexDataset(test_data_words, vocabulary, max_words=MAX_MESSAGE_LENGTH_WORDS)
    train_loader = DataLoader(train_data,
                              batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=IndexVectorCollator())
    test_loader = DataLoader(test_data,
                             batch_size=TEST_BATCH_SIZE,
                             shuffle=True,
                             num_workers=2,
                             collate_fn=IndexVectorCollator())

    writer = SummaryWriter(log_file, purge_step=epoch, flush_secs=60)

    sample_input, sample_lens, _ = next(iter(train_loader))
    summary(model=model, input_data=sample_input.cuda(), lens=sample_lens, device=torch.device('cuda'))

    print("Learning started")

    while epoch < epochs:
        epoch += 1
        print(f"Epoch: {epoch}")
        epoch_losses = []
        epoch_accuracy = []
        model.train()

        loss_fn = FocalLoss(alpha=0.5, gamma=2)

        for step, (x, x_len, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            y_pred = model(x, x_len)
            loss_val = loss_fn(y_pred, y)
            accuracy = torch.argmax(y_pred, 1).eq(y).sum().item() / y.shape[0]

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            epoch_losses.append(loss_val.item())
            epoch_accuracy.append(accuracy)
            print('    Batch {} of {} loss: {}, accuracy: {}, lr: {}'
                  .format(step + 1, len(train_loader), loss_val.item(), accuracy, optimizer.param_groups[0]["lr"]),
                  file=sys.stderr)
        print(f'Train loss: {np.mean(epoch_losses):.4f}, accuracy: {np.mean(epoch_accuracy):.4f}')
        writer.add_scalar('Loss/train', np.mean(epoch_losses), global_step=epoch)
        writer.add_scalar('Accuracy/train', np.mean(epoch_accuracy), global_step=epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]["lr"], global_step=epoch)

        score = evaluate(model, test_loader, loss_fn, writer=writer, epoch=epoch)
        if score > best_score:
            best_model = copy.deepcopy(model)
            best_score = score
            print('New best score')
            save_train_state(epoch, model, optimizer, scheduler, best_score, vocabulary, checkpoint_file)
        scheduler.step()
    if best_score < 0:
        best_score = evaluate(model, test_loader, writer=writer)

    writer.close()
    save_file_path = os.path.join(SAVED_MODELS_PATH,
                                  '{}.{}.{}.{:.2f}.pck'.format(model.__class__.__name__, str(train_data_words),
                                                               datetime.datetime.now().isoformat(),
                                                               best_score))
    log_file_path = os.path.join(LOGS_DIR, '{}.{}.{}.{:.2f}'.format(model.__class__.__name__, str(train_data_words),
                                                                    datetime.datetime.now().isoformat(),
                                                                    best_score))
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    shutil.move(checkpoint_file, save_file_path)
    shutil.move(log_file, log_file_path)

    return best_model, best_score


def train(epochs, dataset_name: str):
    print(f'Training model for {epochs} epochs')
    train_data_words = PreCachedDataset(os.path.join(DATA_DIRECTORY, f'{dataset_name}.train.pck'))
    test_data_words = PreCachedDataset(os.path.join(DATA_DIRECTORY, f'{dataset_name}.test.pck'))

    model = SpamClassifier(VOCABULARY_SIZE)
    train_model(train_data_words, test_data_words, model, epochs)


if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='Train models.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=25, help='How many epochs to train for')
    parser.add_argument('--dataset', type=str, default='', required=True, help='Name of the dataset to use')
    args = parser.parse_args()

    train(args.epochs, args.dataset)
