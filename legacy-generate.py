import mupo
import torch

import pretty_midi 
import argparse
import alive_progress 

import pypianoroll as pr

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, nargs="?", choices=["midi"], default="midi")
    parser.add_argument("-mp", "--model-path", type=str, required=True)
    parser.add_argument("-ip", "--input-path", type=str, required=True)
    parser.add_argument("-op", "--output-path", type=str, required=True)

    parser.add_argument("-i", "--iterations", type=int, required=True)
    parser.add_argument("-cl", "--context-length", type=int, required=True)

    parser.add_argument("-mt", "--model-type", type=str, nargs="?", choices=["kappa", "kappa-ii", "lambda", "mu", "nu", "omicron", "sigma", "sigma-iii"], required=True)

    parser.add_argument("-t", "--temperature", type=float, nargs="?", default=1.0)
    
    args = parser.parse_args()

    mode = args.mode
    model_path = args.model_path
    input_path = args.input_path
    output_path = args.output_path
    iterations = args.iterations
    model_type = args.model_type
    context_length = args.context_length

    temperature = args.temperature
    
    print("configuration:")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    del parser, args

    model = torch.load(model_path, weights_only=False)

    piece_data = None

    match mode:
        case "midi":
            midi_data = pretty_midi.PrettyMIDI(input_path)
            piece_data = mupo.convert_midi_to_tensor(midi_data)
            piece_data = piece_data[:context_length, :]

            del midi_data

    piece_data = piece_data.cuda()

    if model_type == "kappa-ii" or model_type == "mu" or model_type == "nu" or model_type == "omicron" or model_type == "sigma" or model_type == "sigma-iii":
        piece_data = piece_data.unsqueeze(dim=0)

    model = model.cuda()

    model.eval()

    #print(piece_data.shape)

    with alive_progress.alive_bar(iterations, title="iterations", spinner=None) as bar:
        for iteration in range(0, iterations):
            with torch.no_grad():
                model_output = None
                extra = None

                match model_type:
                    case "kappa-ii":
                        model_output = model(piece_data)
                    case "kappa":
                        model_output = model(piece_data)
                    case "lambda":
                        model_output = model(piece_data, True)
                    case "mu":
                        model_output = model(piece_data)
                    case "nu":
                        input_data = piece_data[:, -model.length:, :]
                        model_output = model(input_data, temperature)
                    case "omicron":
                        input_data = piece_data[:, -model.meta_data.mw_size:, :]
                        model_output = model(input_data)
                    case "sigma":
                        model_output = model(piece_data)
                    case "sigma-iii":
                        model_output, extra = model(piece_data)

                #print(piece_data, model_output)

                match model_type:
                    case "kappa":
                        model_output = model_output[-1, :].unsqueeze(dim=0)
                    case "kappa-ii":
                        model_output = model_output.squeeze()
                        model_output = model_output[-1]
                        model_output[0] = torch.round(model_output[0])
                        model_output[1:] = torch.round(model_output[1:], decimals=4)
                        piece_data = piece_data.squeeze()
                    case "mu":
                        model_output = model.format(model_output)
                        model_output = model.apply_transformation(model_output, piece_data)

                        model_output = model_output.squeeze()
                        piece_data = piece_data.squeeze()
                    case "nu":
                        model_output = model.format(model_output)
                        model_output = model.apply_transformation(model_output, piece_data)

                        model_output = model_output.squeeze()
                        piece_data = piece_data.squeeze()
                    case "omicron":
                        model_output = model_output.squeeze()
                        piece_data = piece_data.squeeze()
                    case "sigma":
                        model_output = model_output[0].cpu().detach()
                        values = mupo.sigma_to_values(model_output)
                        #print("VALUES:", values)
                        pitch = values[-1]
                        start = piece_data[0, -1, 1] + 0.2
                        start = start.cpu().detach()
                        duration = torch.tensor(0.2)

                        #print(pitch)
                        #print(start)
                        #print(duration)

                        model_output = torch.stack((pitch, start, duration)).cuda()

                        piece_data = piece_data.squeeze()
                    case "sigma-iii":
                        model_output = model_output[0].cpu().detach()
                        extra = extra[0].cpu().detach()

                        values = mupo.sigma_to_values(model_output)
                        #print("VALUES:", values)
                        pitch = values[-1]
                        start = torch.relu(extra[-1, 0])
                        start = start.cpu().detach()
                        duration = torch.relu(extra[-1, 1])

                        #print(pitch)
                        #print(start)
                        #print(duration)

                        model_output = torch.stack((pitch, start, duration)).cuda()
                        piece_data = piece_data.squeeze()
                        #print(model_output.shape)
                        #print(piece_data.shape)

                        #piece_data = torch.cat((piece_data.squeeze()[0].unsqueeze(dim=0), model_output))


                #print(model_output, piece_data)

                #if model_type != "sigma-iii":
                piece_data = torch.vstack((piece_data, model_output))
   
                if model_type == "kappa-ii" or model_type == "mu" or model_type == "nu" or model_type == "omicron" or model_type == "sigma" or model_type == "sigma-iii":
                    piece_data = piece_data.unsqueeze(dim=0)

                del model_output
    
            bar()

    piece_data = piece_data.cpu()

    #print(piece_data.shape)

    if model_type == "kappa-ii" or model_type == "mu" or model_type == "nu" or model_type == "omicron" or model_type == "sigma" or model_type == "sigma-iii":
        piece_data = piece_data.squeeze()

    match mode:
        case "midi":
            midi_data = mupo.convert_tensor_to_midi(piece_data)

            piano_roll = pr.from_pretty_midi(midi_data)
            piano_roll.plot()

            plt.show()

            midi_data.write(output_path)

            del midi_data


if __name__ == "__main__":
    main()
