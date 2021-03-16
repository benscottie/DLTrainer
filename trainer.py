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
        self.exp_nm = cfg.EXP_NM
        self.exp_dir = cfg.EXP_DIR
        self.exp_path = os.path.join(self.exp_dir, self.exp_nm)
        self.exp_prefix = datetime.today().strftime('%Y-%m-%d')
        self.exp_subdir = self.exp_path
        self.cp_subdir = os.path.join(self.exp_path, cfg.TRAIN.CP_DIR)
        """
        self.exp_subdir = os.path.join(
            self.exp_path,
            self.exp_prefix
        )
        """
        # Data Parameters
        self.model_path = cfg.TRAIN.MODEL.MODEL_PATH
        
        self.train_path = cfg.DATASETS.TRAIN.CSV_PATH
        self.val_path = cfg.DATASETS.VAL.CSV_PATH
        self.test_path = cfg.DATASETS.TEST.CSV_PATH

        self.train_bs = cfg.DATALOADER.TRAIN_BATCH_SIZE
        self.val_bs = cfg.DATALOADER.VAL_BATCH_SIZE
        self.test_bs = cfg.DATALOADER.TEST_BATCH_SIZE

        # Get Data
        self.train_dataset = get_dataset(self.train_path, self.model_path)
        self.train_loader = get_dataloader(self.train_dataset, self.train_bs)

        self.val_dataset = get_dataset(self.val_path, self.model_path)
        self.val_loader = get_dataloader(self.val_dataset, self.val_bs)

        self.test_dataset = get_dataset(self.val_path, self.model_path)
        self.test_loader = get_dataloader(self.test_dataset, self.test_bs)

        # Training Parameters
        self.global_step = 0
        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        #self.train_steps = cfg.TRAIN.TRAIN_STEPS
        self.train_steps = self.num_epochs * len(self.train_loader)
        self.grad_accum_steps = cfg.TRAIN.GRAD_ACCUM_STEPS
        self.eval_steps = cfg.TRAIN.EVAL_STEPS
        self.lr = cfg.TRAIN.OPTIMIZER.LR
        self.scheduler_name = cfg.TRAIN.LR_SCHEDULER.NAME
        self.warmup_steps = cfg.TRAIN.LR_SCHEDULER.WARMUP_STEPS
        self.save_cp = cfg.TRAIN.SAVE_CP
        self.eval_only = cfg.TRAIN.EVAL_ONLY
        
        # Load Model & Optimizer
        self.model = get_model(self.model_path)
        self.model.to(self.device)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_lr_scheduler()
        
        # Results
        self.best_eval_loss = float('inf')
        self.best_cp = None
        self.df_results = pd.DataFrame(columns=['step', 'train_loss', 'train_accuracy', 'running_loss', 'running_accuracy',
                                                'val_loss', 'val_accuracy', 'total_time'])
        self.writer = self._get_writer()
    
    def _make_exp_dir(self):
        try:
            os.mkdir(self.exp_dir)
        except OSError:
            pass
    
    def _make_exp_subdir(self):
        try:
            os.mkdir(self.exp_subdir)
        except OSError:
            pass

    def _make_cp_subdir(self):
        try:
            os.mkdir(self.cp_subdir)
        except OSError:
            pass

    def _get_writer(self):
        log_dir = os.path.join(
            self.exp_subdir,
            f'{self.exp_prefix}-{self.exp_nm}'
        )
        return SummaryWriter(log_dir=log_dir)
    
    def _get_device(self, cfg):
        if cfg.TRAIN.DEVICE == 'gpu' and torch.cuda.is_available():
            dname = 'cuda'
        else:
            dname = 'cpu'
        device = torch.device(dname)
        logger.info(f"Device Set: {device}")
        return device

    def _get_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.lr)
    
    def _get_lr_scheduler(self):
        return get_scheduler(name=self.scheduler_name, optimizer=self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.train_steps)
    
    def _compute_metrics(self, n_correct, labels):
        accuracy = n_correct/len(labels)
        return accuracy
    
    def train(self):

        start_time = time.time()

        for epoch in range(self.num_epochs):

            running_loss = 0
            running_accuracy = 0

            tqdm_desc =  f"Train Epoch {epoch}/{self.num_epochs}"
            with tqdm(self.train_loader, desc=tqdm_desc, unit='batch') as pbar:

                for batch in pbar:

                    self.global_step += 1

                    # Move tensors to the configured device
                    input_ids = batch['text']['input_ids'].to(self.device)
                    token_type_ids = batch['text']['token_type_ids'].to(self.device)
                    attention_mask = batch['text']['attention_mask'].to(self.device)
                    
                    labels = batch['labels'].to(self.device)

                    # Forward pass
                    outputs = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
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
                    pbar.set_postfix({'train_loss': loss.item(), 'train_accuracy': accuracy, 'lr': self.scheduler.get_lr()[0]})

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
                    if (self.global_step/self.grad_accum_steps) % self.eval_steps == 0:
                        results = self._validate_and_save_cp(loss, accuracy, running_loss, running_accuracy, start_time)
                        # Update Tensorboard
                        self.writer.add_scalar('running_loss', running_loss/self.eval_steps, self.global_step)
                        self.writer.add_scalar('running_accuracy', running_accuracy/self.eval_steps, self.global_step)
                        running_loss = 0
                        running_accuracy = 0
                        
        return results
        
        
    def evaluate(self, loader, phase):
            
        n_batches = len(loader)
        running_loss = 0
        running_accuracy = 0
        
        with torch.no_grad():
            tqdm_desc =  f"Running {phase}"
            with tqdm(loader, desc=tqdm_desc, unit='batch') as pbar:
                for batch in pbar:

                    # Move tensors to the configured device
                    input_ids = batch['text']['input_ids'].to(self.device)
                    token_type_ids = batch['text']['token_type_ids'].to(self.device)
                    attention_mask = batch['text']['attention_mask'].to(self.device)
                    
                    labels = batch['labels'].to(self.device)

                    # Forward pass
                    outputs = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
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
                    pbar.set_postfix({f'{phase}_loss': running_loss, f'{phase}_accuracy': running_accuracy})

        # Update Tensorboard
        self.writer.add_scalar(f'{phase}_accuracy', running_accuracy/n_batches, self.global_step)
        self.writer.add_scalar(f'{phase}_loss', running_loss/n_batches, self.global_step)

        # Results
        results = ({'step': self.global_step,
                    'val_loss': val_loss/n_batches,
                    'val_accuracy':val_accuracy/n_batches,
                   })

        return results, pbar


    def _validate_and_save_cp(self, train_loss, train_accuracy, running_loss, running_accuracy, start_time):
        
        # Get evaluation results
        results, pbar = self.evaluate(self.val_loader, 'val')  

        # Add Training Results
        results['train_loss'] = train_loss.item()
        results['train_accuracy'] = train_accuracy
        results['running_loss'] = running_loss/self.eval_steps
        results['running_accuracy'] = running_accuracy/self.eval_steps
        results['total_time'] = time.time() - start_time
  
        self.df_results = self.df_results.append(results, ignore_index=True)
    
        # Log Results
        pbar.write(f"Validation Results: Checkpoint {results['step']}, Time Elapsed {results['total_time']:.2f}")
        pbar.write(f"Training Loss: {results['train_loss']:.3f}, Running Train Loss: {results['running_loss']:.3f}, "
                    f"Validation Loss: {results['val_loss']:.3f}, Validation Accuracy: {results['val_accuracy']*100:.2f}%")
        
        # Save Model Checkpoint and Results
        if self.save_cp == True:
            results_path = os.path.join(self.exp_subdir, f'{self.exp_prefix}-experiment_results.csv')
            self.df_results.to_csv(results_path, index=False) #save results
            model_path = os.path.join(self.cp_subdir, f'{self.exp_prefix}-checkpoint-{self.global_step}')
            self.model.save_pretrained(model_path) #save model
            tqdm.write(f"Saved Checkpoint {results['step']} to {self.exp_dir}")

        # Update best model weights
        if results['val_loss'] < self.best_eval_loss:
            self.best_eval_loss = results['val_loss']
            self.best_cp_name = self.global_step
            model_path = os.path.join(self.cp_subdir, f'{self.exp_prefix}-best_checkpoint')
            self.model.save_pretrained(model_path)
            tqdm.write(f"New Best Validation Checkpoint: {results['step']}")
        
        return results
    
    def run(self):
        
        # make experiment folders
        self._make_exp_dir()
        self._make_exp_subdir()
        self._make_cp_subdir()
        
        results = self.train() # train & validate model
        
        logger.info(f"Evaluating Model Performance on Test Data")
        results, pbar = self.evaluate(self.test_loader, 'test') # evaluate performance on test dataset
        pbar.write(f"Test Results: Time Elapsed {results['total_time']:.2f}")
        pbar.write(f"Test Loss: {results['test_loss']:.3f}, Test Accuracy: {results['test_accuracy']*100:.2f}%")
        
        # Save Final Results
        df_results['model'] = str(self.model_path)
        df_results['lr'] = float(self.lr)
        df_results['num_epochs'] = int(self.num_epochs)
        df_results['train_batch_size'] = int(self.train_bs)
        df_results['learning_rate_scheduler'] = str(self.scheduler_name)
        df_results['warmup_steps'] = int(self.warmup_steps)
        self.df_results = self.df_results.append(results, ignore_index=True)
        results_path = os.path.join(self.exp_subdir, f'{self.exp_prefix}-experiment_results.csv')
        self.df_results.to_csv(results_path, index=False)
        
        # Update TensorBoard
        self.writer.add_hparams({'model': str(self.model_path), 'lr': float(self.lr), 'num_epochs': int(self.num_epochs), 
                                 'train_batch_size': int(self.train_bs), 'learning_rate_scheduler': str(self.scheduler_name), 
                                 'warmup_steps': int(self.warmup_steps)}, 
                                {'hparam/test_loss': results['test_loss'], 'hparam/test_accuracy': results['test_accuracy'],
                                 'hparam/val_loss': results['val_loss'], 'hparam/val_accuracy': results['val_accuracy'],
                                 'hparam/train_loss': results['train_loss'], 'train_accuracy': results['train_accuracy']})
    
    def run_eval(self):
        
        # make experiment folders
        self._make_exp_dir()
        self._make_exp_subdir()
        
        # Evaluate performance on test dataset
        results, pbar = self.evaluate(self.test_loader, 'test') # evaluate performance on test dataset
        pbar.write(f"Test Results: Time Elapsed {results['total_time']:.2f}")
        pbar.write(f"Test Loss: {results['test_loss']:.3f}, Test Accuracy: {results['test_accuracy']*100:.2f}%")
        
        # Save Final Results
        df_results['model'] = str(self.model_path)
        df_results['lr'] = float(self.lr)
        df_results['num_epochs'] = int(self.num_epochs)
        df_results['train_batch_size'] = int(self.train_bs)
        df_results['learning_rate_scheduler'] = str(self.scheduler_name)
        df_results['warmup_steps'] = int(self.warmup_steps)
        self.df_results = self.df_results.append(results, ignore_index=True)
        results_path = os.path.join(self.exp_subdir, f'{self.exp_prefix}-experiment_evaluation.csv')
        self.df_results.to_csv(results_path, index=False)
        
        # Update TensorBoard
        self.writer.add_hparams({'model': str(self.model_path), 'lr': float(self.lr), 'num_epochs': int(self.num_epochs), 
                                 'train_batch_size': int(self.train_bs), 'learning_rate_scheduler': str(self.scheduler_name), 
                                 'warmup_steps': int(self.warmup_steps)}, 
                                {'hparam/test_loss': results['test_loss'], 'hparam/test_accuracy': results['test_accuracy']})
        

def get_parser():
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('--cfg', type=str, help='config file')
    parser.add_argument('--exp_runner_cfg', type=str, help='experiment runner config file')
    return parser

if __name__ == '__main__':

    # Load args and config
    args = get_parser().parse_args()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.exp_runner_cfg:
        cfg.merge_from_file(args.exp_runner_cfg)
    cfg.freeze()

    # Train model
    trainer = Trainer(cfg)
    logger.info(f"Trainer Initialized, Data Loaded")
    
    if cfg.TRAIN.EVAL_ONLY == True: # evaluate on test dataset only
        st = time.time()
        logger.info(f"Beginning Evaluation")
        trainer.run_eval()
        logger.info(f"Evaluation Complete, Time Elapsed {(time.time() - st):.2f}")
    else: # train and validate model
        st = time.time()
        logger.info(f"Beginning Training")
        trainer.run()
        logger.info(f"Training Complete, Time Elapsed {(time.time() - st):.2f}")



        



                        












