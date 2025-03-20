import os
import sys

import argparse
import alive_progress as ap

import mupo

import torch

def main():
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("-m", "--model", type=str, nargs=1, help="path of the model to be trained", required=True)
    parser.add_argument("-d", "--datasets", type=str, nargs="+", help="path of the datasets used in training", required=True)
    parser.add_argument("-sl", "--sequence-lengths", type=int, nargs="+", help="sequence lengths to train the model on")

    parser.add_argument("-bs", "--batch-size", type=int, nargs="?", help="batch size used in training", default=64)
    parser.add_argument("-e", "--epochs", type=int, nargs="?", help="number of epochs in training", default=500)

    parser.add_argument("-v", "--verbosity", type=str, nargs="?", default="normal", choices=["quite", "normal", "verbose", "debug"])

    args = parser.parse_args()

    del parser

    verbosity = args.verbosity

    if verbosity == "debug":
        print(args)

    model_path = args.model[0]
    dataset_paths = args.datasets
    seq_lens = args.sequence_lengths
    batch_size = args.batch_size
    n_epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbosity == "verbose" or verbosity == "debug":
        print(f"device: {device}")

    del args
    
    datasets = []

    if verbosity == "quite":
        for dataset_path in dataset_paths:
            for seq_len in seq_lens:
                data = torch.load(dataset_path)
                dataset = mupo.SequenceDataset(data, seq_len, device)
                datasets.append(dataset)

                del data
                del dataset
            del seq_len
        del dataset_path
    else:
        with ap.alive_bar(len(dataset_paths), title="loading datasets", spinner=None) as bar:
            for dataset_path in dataset_paths:
                for seq_len in seq_lens:
                    data = torch.load(dataset_path)
                    dataset = mupo.SequenceDataset(data, seq_len, device)
                    datasets.append(dataset)

                    del data
                    del dataset
                bar()
                del seq_len
            del dataset_path
        del bar

    dataset = torch.utils.data.ConcatDataset(datasets)

    del datasets
    del dataset_paths
    del seq_lens

    model = torch.load(model_path, weights_only=False).to(device)

    n_epochs = n_epochs

    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.9)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if verbosity == "quite":
        mupo.train(model, dataloader, n_epochs, criterion, optimizer, scheduler)
    else:
        with ap.alive_bar(n_epochs, title="training", spinner=None) as bar:
            mupo.train(model, dataloader, n_epochs, criterion, optimizer, scheduler, out_stream=sys.stdout, alive_bar=bar)

    torch.save(model, model_path)

    return

if __name__ == "__main__":
    main()
