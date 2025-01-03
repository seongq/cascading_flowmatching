import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.odes import ODERegistry
from flowmse.model import VFModel_Finetuning

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     parser = ArgumentParser()
     parser.add_argument("--batch_size", type=int, default=2,  help="During training take at least N_min reverse steps")
     parser.add_argument("--pre_ckpt", type=str,  help="Load ckpt")     
     parser.add_argument("--base_dir", type=str)
     parser.add_argument("--weight_shat",type=float, default=0.8)

     args = parser.parse_args()
     checkpoint_file = args.pre_ckpt
     dataset= os.path.basename(os.path.normpath(args.base_dir))
     
    # Load score model
     model = VFModel_Finetuning.load_from_checkpoint(
        checkpoint_file, base_dir=args.base_dir, weight_shat = args.weight_shat,
        batch_size=args.batch_size, num_workers=4, kwargs=dict(gpu=False)
    )
     model.add_para(args.weight_shat)
     
     from datetime import datetime

     # 현재 날짜와 시간 가져오기
     now = datetime.now()

     # '-'와 ':' 없이 모든 항목을 붙여서 출력하기 (년월일시분초)
     formatted_time = now.strftime("%Y%m%d%H%M%S")
     save_dir_path = f"{model.ode.__class__.__name__}_weight_shat_{args.weight_shat}_dataset_{dataset}_{formatted_time}"
     if model.ode.__class__.__name__ == "FLOWMATCHING":
          logger = WandbLogger(project=f"{model.ode.__class__.__name__}_FINETUNING_weight",  save_dir="logs", name=save_dir_path)
     
     else:
          raise ValueError(f"{model.ode.__class__.__name__}에 대한 configuration이 만들어지지 않았음")
     logger.experiment.log_code(".")

     # Set up callbacks for logger
     callbacks = [ModelCheckpoint(dirpath=f"logs/{save_dir_path}", save_last=True, filename='{epoch}-last')]
     checkpoint_callback_last = ModelCheckpoint(dirpath=f"logs/{save_dir_path}",
          save_last=True, filename='{epoch}-last')
     checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"logs/{save_dir_path}", 
          save_top_k=2, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
     checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"logs/{save_dir_path}", 
          save_top_k=2, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
     #callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr] 
     callbacks = [checkpoint_callback_last, checkpoint_callback_pesq, checkpoint_callback_si_sdr]

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer(  accelerator='gpu', strategy=DDPPlugin(find_unused_parameters=False), gpus=[2,3], auto_select_gpus=False, 
          logger=logger, log_every_n_steps=10, num_sanity_val_steps=1, max_epochs=1000,
          callbacks=callbacks)

     # Train model
     trainer.fit(model)

   