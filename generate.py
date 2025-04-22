import mupo
import torch

import pretty_midi 
import argparse
import alive_progress 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, nargs="?", choices=["midi"], default="midi")
    parser.add_argument("-mp", "--model-path", type=str, required=True)
    parser.add_argument("-ip", "--input-path", type=str, required=True)
    parser.add_argument("-op", "--output-path", type=str, required=True)

    parser.add_argument("-i", "--iterations", type=int, required=True)

    parser.add_argument("-mt", "--model-type", type=str, nargs="?", choices=["kappa", "lambda"], required=True)
    
    args = parser.parse_args()

    mode = args.mode
    model_path = args.model_path
    input_path = args.input_path
    output_path = args.output_path
    iterations = args.iterations
    model_type = args.model_type
    
    print("configuration:")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    del parser, args

    model = torch.load(model_path, weights_only=False)

    piece_data = None

    match mode:
        case "midi":
            midi_data = pretty_midi.PrettyMIDI(input_path)
            piece_data = mupo.convert_midi_to_tensor(midi_data)

            del midi_data

    piece_data = piece_data.cuda()
    model = model.cuda()

    model.eval()

    with alive_progress.alive_bar(iterations, title="iterations", spinner=None) as bar:
        for iteration in range(0, iterations):
            with torch.no_grad():
                model_output = None

                match model_type:
                    case "kappa":
                        model_output = model(piece_data)
                    case "lambda":
                        model_output = model(piece_data, True)

                print(piece_data, model_output)

                if model_type == "kappa":
                    model_output = model_output[-1, :].unsqueeze(dim=0)
    
                piece_data = torch.vstack((piece_data, model_output))
    
                del model_output
    
            bar()

    piece_data = piece_data.cpu()

    match mode:
        case "midi":
            midi_data = mupo.convert_tensor_to_midi(piece_data)
            midi_data.write(output_path)

            del midi_data


if __name__ == "__main__":
    main()
