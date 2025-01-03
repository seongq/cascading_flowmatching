import os
main_folder = "/workspace/flowmatching_weight_condition/flowmse/logs"
folder_directories = os.listdir(main_folder)
folder_directories = [f for f in folder_directories if "SGMSE_CRP" in f and "to_conditioned" in f]
# print(folder_directories)

N_epsilon_values = [name.split('_N_eps_')[1].split('_')[0] for name in folder_directories]
N_eps = 1
Ns = [1,2,3,4,5]
modes = ['mixture']
# modes = ['self', 'mixture']
scripts = []
script = ""
for mode in modes:
    for N in Ns:
        
        for i in range(len(folder_directories)):
            folder = folder_directories[i]
            model_sup_path = os.path.join(main_folder, folder)
            model_path = os.listdir(model_sup_path)[0]
            model_path = os.path.join(main_folder, model_sup_path, model_path)
            if "VCTK_corpus" in model_path:
                dataset = "/data/dataset/VCTK_corpus"
            elif "WSJ0-CHiME3" in model_path:
                dataset = "/data/dataset/WSJ0-CHiME3"
            else:
                raise("에러")
            data_name = dataset.split("/")[-1]
            
            if mode=="self":
                script = f"python evaluate_easy_with_sgmsecrp.py --ckpt {model_path} --N {N} --N_eps {N_eps} --test_dir {dataset}  --condition {mode} --folder_destination sgmse_crp_N_eps_{N_epsilon_values[i]}_to_condition_condition_{mode}_N_{N}_N_eps_{N_eps}_dataset_{data_name}"
            elif mode=="mixture":
                if data_name=="VCTK_corpus" and N_epsilon_values[i]==3 and N==3:
                    script = f"python evaluate_easy_with_sgmsecrp.py --ckpt {model_path} --N {N} --N_eps {0} --test_dir {dataset}  --condition {mode} --folder_destination sgmse_crp_N_eps_{N_epsilon_values[i]}_to_condition_condition_{mode}_N_{N}_dataset_{data_name}"
            
                
            else:
                raise("error")
            if script in scripts:
                continue
            else:
                scripts.append(script)
            
            print(script)
            os.system(script)