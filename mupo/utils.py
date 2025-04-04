import torch

import os
import shutil

import alive_progress as ap

def sequence_length(x):
    if len(x.shape) == 3: # batched
        return x.shape[1]
    elif len(x.shape) == 2: # unbatched
        return x.shape[0]
    else:
        raise RuntimeError(f"Invalid sequence tensor with shape {x.shape}.")


def is_batched(x):
    if len(x.shape) == 3:
        return True
    elif len(x.shape) == 2:
        return False
    else:
        raise RuntimeError(f"Invalid sequence tensor with shape {x.shape}.")


def learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def copy_files(src_dir, dest_dir, filter_substr = None, with_p_bar=False):
    fnames = os.listdir(src_dir)

    if filter_substr is not None:
        fnames = [fn for fn in fnames if fn.find(filter_substr) != -1]

    if not with_p_bar:
        for fn in fnames:
            src_path = os.path.join(src_dir, fn)
            dest_path = os.path.join(dest_dir, fn)

            shutil.copy(src_path, dest_path)

        return
    
    with ap.alive_bar(len(fnames), title="copying files", spinner=None) as bar:
        for fn in fnames:
            src_path = os.path.join(src_dir, fn)
            dest_path = os.path.join(dest_dir, fn)

            shutil.copy(src_path, dest_path)
    
            bar()


def directory_file_names_and_paths(directory):
    return [
        (file_name, os.path.join(directory, file_name))
        for file_name in os.listdir(directory)
    ]
