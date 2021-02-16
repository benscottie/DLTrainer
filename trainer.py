# Libraries
from model import get_model
from dataset import get_dataset, get_dataloader
from transformers import AdamW
from transformers import get_scheduler

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import time
import os
from tqdm import tqdm
import argparse
import logging
from config import get_cfg_defaults
import pandas as pd

# Initialize Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

class Trainer():
    def __init__(self, cfg):

        # Save experiment info
        self.device = self._get_device(cfg)
        self.cfg = cfg
        self.exp_nm = cfg.EXP_NM
        self.exp_dir = cfg.EXP_DIR
        self.exp_path = os.path.join(self.exp_dir, self.exp_nm)
        self.exp_prefix = (datetime.now())
        self.exp_subdir = self.exp_path
        self.cp_subdir = os.path.join(self.exp_path, self.cfg.CP_DIR)
        """
        self.exp_subdir = os.path.join(
            self.exp_path,
            self.exp_prefix
        )
        """
        
        # Load Model & Optimizer
        self.model_path = cfg.TRAIN.MODEL.MODEL_PATH
        self.model = get_model(self.model_path)
        self.model.to(self.device)
        self.optimizer = self._get_optimizer(cfg)
        self.scheduler = self._get_lr_scheduler(cfg)
        self.tokenizer = self._get_tokenizer(cfg)

        # Data Parameters
        self.train_path = cfg.TRAIN.CSV_PATH
        self.val_path = cfg.VAL.CSV_PATH
        self.test_path = cfg.TEST.CSV_PATH

        self.train_bs = cfg.DATALOADER.TRAIN_BATCH_SIZE
        self.val_bs = cfg.DATALOADER.VAL_BATCH_SIZE
        self.test_bs = cfg.DATALOADER.TEST_BATCH_SIZE

        # Get Data
        self.train_dataset = get_dataset(self.train_path, self.model_path)
        self.train_loader = get_dataloader(self.train_dataset, self.train_bs)

        self.val_dataset = get_dataset(self.val_path, self.model_path)
        self.val_loader = get_dataloader(self.val_dataset, self.val_bs)

        self.test_dataset = get_dataset(self.val_path, self.model_path)
        if self.test_dataset & self.test_bs:
            self.test_loader = get_dataloader(self.test_dataset, self.test_bs)
        else:
            test_loader = None

        # Training Parameters
        self.global_step = 0
        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        #self.train_steps = cfg.TRAIN.TRAIN_STEPS
        self.train_steps = self.num_epochs * len(self.train_loader)
        self.grad_accum_steps = cfg.TRAIN.GRAD_ACCUM_STEPS
        self.eval_steps = cfg.TRAIN.EVAL_STEPS
        self.lr = cfg.TRAIN.OPTIMIZER.LR
        self.scheduler = cfg.TRAIN.LR_SCHEDULER.NAME
        self.warmup_steps = cfg.TRAIN.LR_SCHEDULER.WARMUP_STEPS

        # Results
        self.best_eval_loss = float('inf')
        self.best_cp = None
        self.df_results = pd.DataFrame(columns=['step', 'batch_loss', 'epoch_loss', 'eval_loss', 'accuracy', 'f1', 'total_time'])
        self.writer = self._get_writer(cfg.WRTIER)
        self.writer.add_params({'model': self.model_path, 'lr': self.lr, 'num_epochs': self.epochs, 'train_batch_size': self.train_bs, 'learning_rate_scheduler': self.scheduler, 'warmup_steps': self.warmup_steps})
    
    def _make_exp_dir(self):
        os.mkdir(self.exp_dir)
    
    def _make_exp_subdir(self):
        os.mkdir(self.exp_subdir)

    def _make_cp_subdir(self):
        os.mkdir(self.cp_subdir)

    def _get_writer(self):
        log_dir = os.path.join(
            self.exp_subdir,
            f'{self.exp_prefix}-{self.exp_nm}'
        )
        return SummaryWriter(log_dir=log_dir)
    
    def _get_device(self):
        if self.cfg.DEVICE == 'gpu' and torch.cuda.is_available():
            dname = 'cuda'
        else:
            dname = 'cpu'
        device = torch.device(dname)
        logger.info(f"Device Set: {device}")
        return device

    def _get_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.lr)
    
    def _get_lr_scheduler(self):
        return get_scheduler(name=self.scheduler, optimizer=self.optimizer, num_warmup_steps=self.WARMUP_STEPS, num_training_steps=self.train_steps)
    
    def _compute_metrics(self, n_correct, labels):
        accuracy = n_correct/len(labels)
        return accuracy
    
    def train(self):

        # make experiment folders
        self._make_exp_dir()
        self._make_exp_subdir()
        self._make_cp_subdir()

        start_time = time.time()

        logger.info("Beginning Training")
        for epoch in range(self.num_epochs):

            running_loss = 0
            running_accuracy = 0

            tqdm_desc =  f'Train Epoch {epoch}/{self.num_epochs}'
            with tqdm(self.train_loader, desc=tqdm_desc, units='batches') as pbar:

                for batch in pbar:

                    self.global_step += 1

                    # Move tensors to the configured device
                    inputs = batch.text.to(self.device)
                    labels = batch.labels.to(self.device)

                    # Forward pass
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    #loss = loss_fn(outputs, labels)
                    running_loss += loss.item()

                    # Track key metrics
                    y_preds = F.softmax(outputs.logits)
                    pred_class = torch.argmax(y_preds)
                    n_correct = torch.sum(pred_class == labels).item()
                    accuracy = self._compute_metrics(n_correct, labels)
                    running_accuracy += accuracy
    
                    # Update progress bar
                    pbar.set_postfix(**{f'train_loss': loss.item()})
                    pbar.set_postfix(**{f'train_accuracy': accuracy})
                    pbar.set_postfix(**{f'lr': self.scheduler.get_lr()[0]})

                    # Update Tensorboard
                    self.writer.add_scalar('train_accuracy', accuracy, self.global_step)
                    self.writer.add_scalar('train_loss', loss.item(), self.global_step)

                    # Backpropagate loss and update optimizer
                    if (self.global_step % self.grad_accum_steps) == 0:
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step() #check
                        self.optimizer.zero_grad() #zero the parameter gradients
                    
                    # Validate at Checkpoint
                    if (self.global_step/self.grad_accum_steps) % cfg.EVAL_STEPS == 0:
                        self.df_results = self._validate_and_save_cp(loss, running_loss, running_accuracy, start_time)
                        # Update Tensorboard
                        self.writer.add_scalar('running_loss', running_loss/self.eval_steps, self.global_step)
                        self.writer.add_scalar('running_accuracy', running_accuracy/self.eval_steps, self.global_step)
                        epoch_loss = 0

    def evaluate(self):

        n_seqs = len(self.val_dataset)
        n_batches = len(self.val_loader)
        val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            tqdm_desc =  f'Running Validation'
            with tqdm(self.val_loader, desc=tqdm_desc, units='batches') as pbar:
                for batch in pbar:

                    # Move tensors to the configured device
                    inputs = batch.text.to(self.device)
                    labels = batch.label.to(self.device)

                    # Forward pass
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    #loss = loss_fn(outputs, labels)
                    val_loss += loss.item()

                    # Track key metrics
                    y_preds = F.softmax(outputs.logits)
                    pred_class = torch.argmax(y_preds)
                    n_correct = torch.sum(pred_class == labels).item()
                    accuracy = self._compute_metrics(n_correct, labels)
                    val_accuracy += accuracy

        # Update progress bar
        pbar.set_postfix(**{f'val_loss': val_loss/n_batches})
        pbar.set_postfix(**{f'val_accuracy': val_accuracy/n_batches})

        # Update Tensorboard
        self.writer.add_scalar('val_accuracy', val_accuracy/n_batches, self.global_step)
        self.writer.add_scalar('val_loss', val_loss/n_batches, self.global_step)

        # Results
        results = ({'step': self.global_step,
                #'train_loss': round(train_loss, 4),
                #'train_accuracy': round(train_accuracy, 2),
                #'running_loss': round((running_loss/self.eval_steps), 4),
                #'running_accuracy': round((running_accuracy/self.eval_steps), 2),
                'val_loss': round((val_loss/n_batches), 4),
                'val_accuracy': round((val_accuracy/n_batches), 4),
                #'total_time': round(time.time() - start_time, 2),
                })

        return results


    def _validate_and_save_cp(self, train_loss, train_accuracy, running_loss, running_accuracy, start_time):
        
        # Get evaluation results
        results = self.evaluate()  

        # Add Training Results
        results['train_loss'] = round(train_loss, 4)
        results['train_accuracy'] = round(train_accuracy, 2)
        results['running_loss'] = round((running_loss/self.eval_steps), 4)
        results['running_accuracy'] = round((running_accuracy/self.eval_steps), 2),
        results['total_time'] = round(time.time() - start_time, 2)

        logger.info(f"Validation Results: Checkpoint {results['step']}, Time Elapsed {results['total_time']:.2f}")
        logger.info(f"Training Loss: {results['train_loss']:.3f}, Running Train Loss: {results['running_loss']:.3f}, Validation Loss: {results['val_loss']:.3f}, Validation Accuracy: {results['val_accuracy']*100:.3f}%")
  

        self.df_results = self.df_results.append(results, ignore_index=True)

        if self.cfg.save_cp == True:
            results_path = os.path.join(self.exp_subdir, f'{self.exp_prefix}-experiment_results.csv')
            self.df_results.to_csv(results_path, index=False) #save results
            model_path = os.path.join(self.cp_subdir, f'{self.exp_prefix}-checkpoint-{self.global_step}.csv')
            self.model.save_pretrained(model_path) #save model
            logger.info(f'Saved Checkpoint {results['global_step']} to {self.exp_dir}')

        # Update best model weights
        if results['eval_loss'] > self.best.eval_loss:
            self.best_eval_loss = results['eval_loss']
            self.best_cp_name = self.global_step
            model_path = os.path.join(self.cp_subdir, f'{self.exp_prefix}-best_checkpoint-{self.global_step}.csv')
            self.model.save_pretrained(model_path)
            logger.info(f'New Best Validation Checkpoint: {results['global_step']')
        
        return self.df_results

def get_parser():
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('--cfg', type=str, help='config file')
    parser.add_argument('--exp_runner_cfg', type=str, help='experiment runner config file')
    return parser

if __name__ == '__main__':

    # Load args and config
    arsgs = get_parser().parse_args()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.exp_runner_cfg:
        cfg.merge_from_file(args.exp_runner_cfg)
    cfg.freeze()

    # Train model
    trainer = Trainer(cfg)
    logger.info(f"Trainer Initialized, Data Loaded")
    if cfg.eval_only == True:
        st = time.time()
        logger.info(f"Beginning Evaluation")
        trainer.evaluate()
        logger.info(f"Evaluation Complete, Time Elapsed {(time.time() - st):2f}")
    else:
        st = time.time()
        logger.info(f"Beginning Training")
        trainer.train()
        logger.info(f"Training Complete, Time Elapsed {(time.time() - st):2f}")



        



                        












