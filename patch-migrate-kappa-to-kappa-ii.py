import torch
import argparse
import mupo

parser = argparse.ArgumentParser()
parser.add_argument("-ip", "--input-path", type=str, required=True)
parser.add_argument("-op", "--output-path", type=str, required=True)

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

if input_path == output_path:
    raise RuntimeError("NEVER USE THE SAME PATH FOR INPUT AND OUTPUT! IF YOU DO THIS AND SOMETHING GOES WRONG THE ORIGINAL WILL BE OVERRIDDEN, AND WILL BE LOST!!!")

# don't get confused by the name! "kappa-ii" refers to 
# what previously was just "kappa". the entire reason why this script exists, is to migrate the two versions i and ii.
kappa_ii = torch.load(input_path, weights_only=False)

new_kappa_ii = mupo.KappaModelII()
new_kappa_ii.load_state_dict(kappa_ii.state_dict())

torch.save(new_kappa_ii, output_path)

print("Done!")
