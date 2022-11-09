import numpy as np
import torch


class EarlyStopping:
    """在给定训练次数的之后验证集的损失没有提高，提前停止训练"""

    def __init__(self, patience=5, verbose=False, delta=0.001, save_path='best.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.001
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.save_path = save_path
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, tokenizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, tokenizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, tokenizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, tokenizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.uie_model.save_pretrained(self.save_path, state_dict=model.state_dict())
        tokenizer.save_pretrained(self.save_path)
        self.val_loss_min = val_loss
