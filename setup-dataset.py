import argparse
import os
import torch

from tqdm import tqdm

import mupo

def create_tau_dataset(load_directory, save_directory, name, filter_fn, progress_bar=True): # filter_fn : bool(str)
    paths = [os.path.join(load_directory, base_name) for base_name in os.listdir(load_directory)]
    paths = [path for path in paths if os.path.splitext(os.path.basename(path))[1] == ".dat"]
    paths = [path for path in paths if filter_fn(os.path.basename(path))]

    iterator = iter(paths)

    if progress_bar:
        iterator = tqdm(iterator, desc="paths", leave=False)

    pieces = []

    for path in iterator:
        piece = torch.load(path, weights_only=False)
        pieces.append(piece)

    raw_dataset = mupo.tau.RawDataset(pieces)

    torch.save(raw_dataset, os.path.join(save_directory, name+".dat"))

def create_ypsilon_dataset(load_directory, save_directory, name, filter_fn, progress_bar=True): # filter_fn : bool(str)
    paths = [os.path.join(load_directory, base_name) for base_name in os.listdir(load_directory)]
    paths = [path for path in paths if os.path.splitext(os.path.basename(path))[1] == ".dat"]
    paths = [path for path in paths if filter_fn(os.path.basename(path))]

    iterator = iter(paths)

    if progress_bar:
        iterator = tqdm(iterator, desc="paths", leave=False)

    pieces = []

    for path in iterator:
        piece = torch.load(path, weights_only=False)
        pieces.append(piece)

    raw_dataset = mupo.tau.RawDataset(pieces)

    torch.save(raw_dataset, os.path.join(save_directory, name+".dat"))


parser = argparse.ArgumentParser()
parser.add_argument("dataset_type", type=str, choices=["tau", "ypsilon"])
parser.add_argument("load_directory", type=str)
parser.add_argument("save_directory", type=str)
parser.add_argument("name", type=str)
parser.add_argument("-f", "--filter_str", type=str, default="", required=False)

parser.add_argument("--no_progress_bar", action="store_false", required=False)

args = parser.parse_args()

dataset_type = args.dataset_type
load_directory = args.load_directory
save_directory = args.save_directory
name = args.name
filter_str = args.filter_str
use_filter = (filter_str != "")

progress_bar = args.no_progress_bar

match dataset_type:
    case "tau":
        filter_fn = lambda x: filter_str in x if use_filter else lambda x: True

        create_tau_dataset(load_directory, save_directory, name, filter_fn, progress_bar)
    case "ypsilon":
        filter_fn = lambda x: filter_str in x if use_filter else lambda x: True

        create_ypsilon_dataset(load_directory, save_directory, name, filter_fn, progress_bar)
