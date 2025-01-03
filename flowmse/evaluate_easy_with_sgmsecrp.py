import time
import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from torchaudio import load
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd

from flowmse.data_module import SpecsDataModule
from flowmse.model import VFModel, VFModel_Finetuning, VFModel_Finetuning_SGMSE_CRP
import pdb
import os
from flowmse.util.other import pad_spec
from flowmse.sampling import get_white_box_solver
from utils import energy_ratios, ensure_dir, print_mean_std

import pdb

if __name__ == '__main__':
    parser = ArgumentParser()
   

    parser.add_argument("--odesolver", type=str,
                        default="euler", help="euler")
    parser.add_argument("--condition", type=str, choices=('mixture', 'self',"clean"))
    parser.add_argument("--reverse_starting_point", type=float, default=1.0, help="Starting point for the reverse SDE.")
    parser.add_argument("--reverse_end_point", type=float, default=0.03)
    
    parser.add_argument("--test_dir")
    parser.add_argument("--folder_destination", type=str, help="Name of destination folder.")    
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--N", type=int, default=5, help="Number of reverse steps")    
    parser.add_argument("--N_eps", type=int, default=1, help="Number of reverse steps")    
    parser.add_argument("--weight_shat",type=float, default=0.8)
    args = parser.parse_args()

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")

    checkpoint_file = args.ckpt
    
   
    target_dir = "/data/{}/".format(
        args.folder_destination)
   
    ensure_dir(target_dir + "files/")

    # Settings
    sr = 16000
    
    odesolver = args.odesolver
    N = args.N
    N_eps = args.N_eps
    
    
    # Load score model
    try:
        model = VFModel_Finetuning.load_from_checkpoint(
            checkpoint_file, base_dir="",
            batch_size=8, num_workers=4, kwargs=dict(gpu=False)
        )
    except:
        model = VFModel_Finetuning_SGMSE_CRP.load_from_checkpoint(
            checkpoint_file, base_dir="", batch_size=8, num_workers=4, kwargs=dict(gpu=False)
        )
    model.weight_shat = args.weight_shat
    reverse_starting_point = args.reverse_starting_point
    reverse_end_point = args.reverse_end_point
    weight_shat = args.weight_shat
    condition = args.condition
        
    model.ode.T_rev = reverse_starting_point
        
    
    model.eval(no_ema=False)
    model.cuda()

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    




    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": []}
    for cnt, noisy_file in tqdm(enumerate(noisy_files)):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        x, _ = load(join(clean_dir, filename))
        y, _ = load(noisy_file)

        #pdb.set_trace()        

         
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        
        
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        Y = Y.cuda()
        
        with torch.no_grad():
            if condition == "mixture":
                Y_prior,_ = model.ode.prior_sampling(Y.shape, Y)
                x_t = Y_prior.to(Y.device)
                #Euler
                timesteps = torch.linspace(reverse_starting_point, reverse_end_point, N, device=Y.device)
                for i in range(len(timesteps)):
                    t = timesteps[i]
                    if i == len(timesteps)-1:
                        dt = 0-t
                    else:
                        dt = timesteps[i+1]-t
                    vect = torch.ones(Y.shape[0], device=Y.device)*t
                    x_t = x_t + dt * model(x_t, vect, Y)            
                
                
            elif condition == "self":
                Y_prior,_ = model.ode.prior_sampling(Y.shape, Y)
                if N ==1 and N_eps ==0:
                    x_t = Y_prior.to(Y.device)
                    vect = torch.ones(Y.shape[0], device=Y.device)
                    dt = 0-1
                    x_t = x_t + dt * model(x_t, vect, Y)
                elif N_eps > 0:
        
        
                    self_condition = Y_prior.to(Y.device)
                    # print(self_condition.device)
                    timesteps = torch.linspace(reverse_starting_point, reverse_end_point, N_eps, device=Y.device)
                    # print(timesteps.device)
                    for i in range(len(timesteps)):
                        t = timesteps[i]
                        if i==len(timesteps)-1:
                            dt = 0 - t
                        else:
                            dt = timesteps[i+1]-t
                    
                        vect = t*torch.ones(Y.shape[0], device=Y.device)
                        # print(vect.device)
                        self_condition = self_condition + dt* model(self_condition, vect,Y)
                    
                    self_condition = (1-weight_shat)*Y.clone() + (weight_shat) * self_condition.clone()
                    x_t = Y_prior.to(Y.device)
                    timesteps = torch.linspace(reverse_starting_point, reverse_end_point, N, device=Y.device)
                    for i in range(len(timesteps)):
                        t = timesteps[i]
                        if i == len(timesteps)-1:
                            dt = 0-t
                        else:
                            dt = timesteps[i+1]-t
                        vect = torch.ones(Y.shape[0], device=Y.device)*t
                        x_t = x_t + dt * model(x_t, vect, self_condition)            
                        
                    #Euler인데 condition 넣어서
                else:
                    raise(f"N 크기가 {N}이다. N을 확인해봐.")
                    
            elif condition=="clean":
                Y_prior,_ = model.ode.prior_sampling(Y.shape, Y)
                X = torch.unsqueeze(model._forward_transform(model._stft(x.cuda())), 0)
                X = pad_spec(X)
                X = X.cuda()
                
                x_t = Y_prior.to(Y.device)
                timesteps = torch.linspace(reverse_starting_point, reverse_end_point, N, device=Y.device)
                for i in range(len(timesteps)):
                    t = timesteps[i]
                    if i == len(timesteps)-1:
                        dt = 0-t
                    else:
                        dt = timesteps[i+1]-t
                    vect = torch.ones(X.shape[0], device=X.device)*t
                    x_t = x_t + dt * model(x_t, vect, X) 
                    print("x_t 연산함")
                
        
        
        sample = x_t.clone()
        
        
        sample = sample.squeeze()
        
        x_hat = model.to_audio(sample, T_orig)
        # print("완료")
        y = y * norm_factor
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        
      
        # Convert to numpy
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        n = y - x

        # Write enhanced wav file
        write(target_dir + "files/" + filename, x_hat, 16000)

        # Append metrics to data frame
        data["filename"].append(filename)
        try:
            p = pesq(sr, x, x_hat, 'wb')
        except: 
            p = float("nan")
        data["pesq"].append(p)
        data["estoi"].append(stoi(x, x_hat, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])

    # Save results as DataFrame
    df = pd.DataFrame(data)
    df.to_csv(join(target_dir, "_results.csv"), index=False)

    # Save average results
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w') as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))

    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        
        file.write("odesolver: {}\n".format(odesolver))
        file.write("weight_shat: {}\n".format(model.weight_shat))
        file.write("weight_y: {}\n".format(model.weight_y))
        
        file.write("N: {}\n".format(N))
        
        file.write("Reverse starting point: {}\n".format(reverse_starting_point))
        file.write("Reverse end point: {}\n".format(reverse_end_point))
        
        file.write("data: {}\n".format(args.test_dir))
        