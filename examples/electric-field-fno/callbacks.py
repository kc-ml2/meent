import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from neuralop.training.callbacks import BasicLoggerCallback, CheckpointCallback, Callback

class TensorboardCallback(BasicLoggerCallback):
    def __init__(self, log_dir='./runs'):
        super().__init__()

        self.writer = SummaryWriter(log_dir=log_dir)
    
    def on_val_end(self, *args, **kwargs):
        print(self.state_dict['msg'])
        sys.stdout.flush()
        
        for pg in self.state_dict['optimizer'].param_groups:
            lr = pg['lr']
            self.state_dict['values_to_log']['lr'] = lr

        for k, v in self.state_dict['values_to_log'].items():
            self.writer.add_scalar(k, v, self.state_dict['epoch'] + 1)

    
class CustomCheckpointCallback(CheckpointCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, *args, **kwargs):
        # Save states to save_dir 
        if self.state_dict['epoch'] % self.save_interval == 0:
            save_path = self.save_dir / f"model{self.state_dict['epoch']}.pt"
            torch.save(self.state_dict['model'].state_dict(), save_path)