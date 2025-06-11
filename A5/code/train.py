import sys
import numpy as np
import datetime
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from get_train_data import FeatureExtractor
from model import BaseModel, WordPOSModel

argparser = argparse.ArgumentParser()
argparser.add_argument('--input_file', default='input_train.npy')
argparser.add_argument('--target_file', default='target_train.npy')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt')
argparser.add_argument('--model', default='wordspos', choices=['base', 'wordspos'],
                       help='path to save model file, if not specified, a .pt with timestamp will be used')

if __name__ == "__main__":
    args = argparser.parse_args()
    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
        rel_vocab_file = open(args.rel_vocab, "r")
    except FileNotFoundError:
        print(f'Could not find vocabulary files {args.words_vocab}, {args.pos_vocab}, and {args.rel_vocab}')
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file, rel_vocab_file)
    word_vocab_size = len(extractor.word_vocab)
    pos_vocab_size = len(extractor.pos_vocab)
    output_size = len(extractor.rel_vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    ### START YOUR CODE ###
    # TODO: Initialize the model
    # Initialize the model
    if args.model == 'base':
        model = BaseModel(word_vocab_size, output_size).to(device)
    else:
        model = WordPOSModel(word_vocab_size, pos_vocab_size, output_size).to(device)
    ### END YOUR CODE ###

    learning_rate = 0.0025
    n_epochs = 100
    batch_size = 256

    # Set optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    inputs = np.load(args.input_file)
    targets = np.load(args.target_file)  # pytorch input is int
    print("Done loading data.")

    # Train loop
    ### START YOUR CODE ###
    # TODO: Wrap inputs and targets into tensors
    # Convert data to tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    ### END YOUR CODE ###
    # Create DataLoader
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(inputs) // batch_size

    loss_history = []  # To store loss values for plotting

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        model.train()
        # for batch in dataloader:
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False):
            ### START YOUR CODE ###
            # TODO: Get inputs and targets from batch; feed inputs to model and compute loss; backpropagate and update model parameters
            inputs_batch, targets_batch = batch
            inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs_batch)
            # Calculate loss
            loss = criterion(outputs, targets_batch)
            loss.backward()  # Backpropagate
            # Update model parameters
            optimizer.step()
            ### END YOUR CODE ###
            epoch_loss += loss.item()

        # Step the scheduler at the end of the epoch
        scheduler.step()

        # print
        print()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        epoch_end_time = time.time()
        print(f'Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.4f}, time: {epoch_end_time - epoch_start_time:.2f} sec')

    # save model
    if args.model is not None:
        torch.save(model.state_dict(), args.model)
    else:
        now = datetime.datetime.now()
        torch.save(model.state_dict(), f'model_{now}.pth')

    # Plot loss curve
    plt.plot(np.arange(1, n_epochs + 1, 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, n_epochs + 1, 10))
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
    plt.savefig(f'../plot/{args.model}_loss_curve_{learning_rate}.png')
