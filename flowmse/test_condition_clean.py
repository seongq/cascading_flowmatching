
import os

Ns = [1,2,3,4,5,6,7,8,9,10]
ckpts = [  "/workspace/flowmatching_20240514/logs/FLOWMATCHING_sigma_min_0.0_sigma_max_0.487_T_rev_1.0_t_eps_0.03_dataset_WSJ0-CHiME3_dor8klbq/last.ckpt",
         "/workspace/flowmatching_20240514/logs/3isssfv0_flowmatching_vctk/epoch=220-pesq=2.66.ckpt",
         "/workspace/flowmatching_weight_condition/flowmse/logs/FLOWMATCHING_weight_shat_0.8_dataset_VCTK_corpus_20240913055146/epoch=0-pesq=2.78.ckpt",
         "/workspace/flowmatching_weight_condition/flowmse/logs/FLOWMATCHING_weight_shat_0.8_dataset_WSJ0-CHiME3_20240919135349/epoch=0-pesq=3.41.ckpt",
         ]
modes = ["clean"]
scripts = []
for N in Ns:
    for ckpt in ckpts:
        print(os.path.exists(ckpt))
        for mode in modes:
            
            
            model_path = ckpt
            if ("VCTK_corpus" in model_path) or "vctk" in model_path:
                dataset = "/data/dataset/VCTK_corpus"
            elif "WSJ0-CHiME3" in model_path:
                dataset = "/data/dataset/WSJ0-CHiME3"
            else:
                raise("에러")
            data_name = dataset.split("/")[-1]
            
            if mode == "clean":
                if "condition" in model_path:
                    folder_destination= f"conditioned_with_1_{mode}_N_{N}_dataset_{data_name}/"
                    folder_destination_check = f"/data/{folder_destination}_avg_results.txt"
                    script = f"CUDA_VISIBLE_DEVICES=2 python evaluate_easy_with_sgmsecrp.py --ckpt {model_path} --N {N} --N_eps {0} --test_dir {dataset}  --condition {mode} --folder_destination {folder_destination}"
                else:
                    folder_destination= f"flowse_{mode}_N_{N}_dataset_{data_name}/"
                    folder_destination_check = f"/data/{folder_destination}_avg_results.txt"
                    script = f"CUDA_VISIBLE_DEVICES=1 python evaluate_easy_with_sgmsecrp.py --ckpt {model_path} --N {N} --N_eps {0} --test_dir {dataset}  --condition {mode} --folder_destination {folder_destination}"
            else:
                continue
            if script in scripts:
                continue
            else:
                scripts.append(script)
                
            if os.path.exists(folder_destination_check):
                    continue
            else:
                pass
            print("===============================================")
            print(script)
            print("================================================")
            os.system(script)
            
                        
