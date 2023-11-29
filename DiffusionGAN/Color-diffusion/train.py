from argparse import ArgumentParser
import wandb
import torch
import pytorch_lightning as pl
from dataset import ColorizationDataset, make_dataloaders
from model import ColorDiffusion
from utils import get_device, load_default_configs
from pytorch_lightning.loggers import WandbLogger
from denoising import Unet, Encoder
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import time

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log", default=False)
    parser.add_argument("--cpu-only", default=False)
    parser.add_argument("--dataset", default="./celeba2", help="Path to unzipped dataset (see readme for download info)")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    print(args)

    enc_config, unet_config, colordiff_config = load_default_configs()
    train_dl, val_dl = make_dataloaders(args.dataset, colordiff_config, num_workers=2, limit=5)
    colordiff_config["sample"] = False
    colordiff_config["should_log"] = args.log
    print("Dataloaders completed")
    #TODO remove 
    # args.ckpt = "/home/ec2-user/Color-diffusion/Color_diffusion_v2/23l96nt1/checkpoints/last.ckpt"
    ckpt_dir = os.listdir("./checkpoints") 
    #args.ckpt = "last.ckpt"
    args.ckpt = None

    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config)
    if args.ckpt is not None:
        print(f"Resuming training from checkpoint: {args.ckpt}")
        model = ColorDiffusion.load_from_checkpoint(
            args.ckpt,
            strict=True, 
            unet=unet, 
            encoder=encoder, 
            train_dl=train_dl, 
            val_dl=val_dl, 
            **colordiff_config
            )
    else:
        model = ColorDiffusion(unet=unet,
                               encoder=encoder, 
                               train_dl=train_dl,
                               val_dl=val_dl, 
                               **colordiff_config)
        print("Model initialized")
    if args.log:
        wandb_logger = WandbLogger(project="Color_diffusion_v2")
        wandb_logger.watch(unet)
        wandb_logger.experiment.config.update(colordiff_config)
        wandb_logger.experiment.config.update(unet_config)
    ckpt_callback = ModelCheckpoint(every_n_train_steps=100, save_top_k=2, save_last=True, monitor="val_loss")
    trainer = pl.Trainer(max_epochs=colordiff_config["epochs"],
                        logger=wandb_logger if args.log else None, 
                        accelerator=colordiff_config["device"],
                        num_sanity_val_steps=30,
                        devices= "auto",
                        log_every_n_steps=50,
                        callbacks=False,
                        profiler="simple" if args.log else None,
                        accumulate_grad_batches=colordiff_config["accumulate_grad_batches"],
                        )
    print("Trainer initialized")
    print(colordiff_config["epochs"])
    #patch_train = pl.trainer.trainer._PatchDataLoader(train_dl)
    #patch_val = pl.trainer.trainer._PatchDataLoader(val_dl)
    trainer.fit(model, train_dl, val_dl)
    
    trainer.save_checkpoint("./checkpoints/20231128_ckpt_200000.ckpt")


