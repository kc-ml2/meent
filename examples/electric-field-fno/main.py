import hydra
import torch

from torch.utils.data import DataLoader

from neuralop import Trainer
from neuralop import LpLoss, H1Loss
from neuralop.utils import count_model_params

import models
from utils import seed_all
from callbacks import TensorboardCallback, CustomCheckpointCallback
from losses import (
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
    RangeAwareLoss,
    LossFuncWrapper,
    CarstenLossFuncWrapper,
    L2pMSSSIM,
    GratingLoss,
    SuperstrateLoss,
    SubstrateLoss,
)


def load_data(data_dir):
    train_ds = torch.load(data_dir)
    test_ds = torch.load(data_dir)
        
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)  # , shuffle=True)

    return train_loader, test_loader


@hydra.main(config_path='.', config_name='config')
def main(cfg):
    seed_all(cfg.seed)

    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model = getattr(models, cfg.model)(**cfg.model_config)
    model = model.to(cfg.device)
    
    train_loader, test_loader = load_data(cfg.train_data_dir, cfg.test_data_dir)

    optimizer = getattr(torch.optim, cfg.optim)(
        model.parameters(), **cfg.optim_config
    )
    
    scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler)(
        optimizer, **cfg.scheduler_config
    )

    loss_dict = {
        'h1': H1Loss(d=2, **cfg.loss_config),
        'l2': LpLoss(d=2, p=2, **cfg.loss_config),
        'ssim': LossFuncWrapper(
            StructuralSimilarityIndexMeasure().to(cfg.device)
        ),
        'msssim': LossFuncWrapper(
            MultiScaleStructuralSimilarityIndexMeasure().to(cfg.device)
        ),
        'l2pmsssim': L2pMSSSIM(cfg.device),
        'carsten_l2': CarstenLossFuncWrapper(
            LpLoss(d=2, p=2, **cfg.loss_config)
        ),
        'carsten_h1': CarstenLossFuncWrapper(
            H1Loss(d=2, **cfg.loss_config)
        ),
        'carsten_lp2msssim': CarstenLossFuncWrapper(
            L2pMSSSIM(cfg.device)
        ),
        'range_aware_l2': RangeAwareLoss(
            LpLoss(d=2, p=2, **cfg.loss_config)
        ),
        'range_aware_h1': RangeAwareLoss(
            H1Loss(d=2, **cfg.loss_config)
        ),
        'grating_l2': GratingLoss(
            LpLoss(d=2, p=2, **cfg.loss_config)
        ),
        'superstrate_l2': SuperstrateLoss(
            LpLoss(d=2, p=2, **cfg.loss_config)
        ),
        'substrate_l2': SubstrateLoss(
            LpLoss(d=2, p=2, **cfg.loss_config)
        ),
    }

    train_loss = loss_dict[cfg.loss] # single loss
    eval_losses = {k:loss_dict[k] for k in cfg.eval_losses} # multiple losses

    print('\n### MODEL ###\n', model)
    print(f'\n### MODEL PARAMS ###\n{count_model_params(model):,}')
    print('\n### EXPERIMENT ###\n', cfg.experiment)
    print('\n### LOSSES ###')
    print(f'\n * Train: {cfg.loss}')
    print(f'\n * Test: {cfg.eval_losses}')
    

    trainer = Trainer(
        model=model, n_epochs=cfg.n_epochs,
        device=cfg.device,
    #   data_processor=data_processor,
        wandb_log=False,
        log_test_interval=cfg.log_test_interval,
        use_distributed=False,
        verbose=True,
        callbacks=[
            TensorboardCallback(log_dir=log_dir), 
            CustomCheckpointCallback(
                save_dir=log_dir, 
                save_interval=cfg.save_interval
            )
        ]
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders={'test': test_loader},
        optimizer=optimizer,
        scheduler=scheduler, 
        regularizer=False, 
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    torch.save(model.state_dict(), f'{log_dir}/model-final.pt')


if __name__ == "__main__":
    main()
