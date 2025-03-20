import os

import argparse

import mupo

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--destination", type=str, nargs=1, required=True, help="destination path of the model")
    parser.add_argument("-t", "--type", type=str, nargs=1, choices=["hyper-model", "fallback-model", "sequence-model"], required=True, help="type of model to create")
    parser.add_argument("-sl", "--sequence-length", type=int, nargs='?', help="when type=sequence-model represent the sequence length of that model")
    
    args = parser.parse_args()

    del parser


    model_path = args.destination[0]
    model_type = args.type[0]
    seq_len = args.sequence_length

    if seq_len is None:
        del seq_len

    del args

    directory = os.path.dirname(model_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    del directory

    match model_type:
        case "hyper-model":
            model = mupo.HyperModel()
            torch.save(model, model_path)
        case "fallback-model":
            model = mupo.FallbackModel()
            torch.save(model, model_path)
        case "sequence-model":
            model = mupo.SequenceModel(seq_len)
            torch.save(model, model_path)
    

if __name__ == "__main__":
    main()
