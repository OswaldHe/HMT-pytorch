from collections import defaultdict
import importlib
import itertools
import logging
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import get_scheduler
from tqdm import tqdm
import horovod.torch as hvd


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, model, optimizer, train_dataloader, valid_dataloader,
                 train_sampler=None, batch_transform_fn=None, get_metrics_fn=lambda _, y: {'loss': y['loss']}) -> None:
        """Implements training loop with horovod multi-gpu & apex fp16 support.

        Args:
            model: torch model to train
            optimizer: torch optimizer
            train_dataloader (torch.utils.data.DataLoader): train set torch dataloader
            valid_dataloader (Optional(torch.utils.data.DataLoader)]): validation set torch dataloader, optional.
            args: params from CLI
        """
        # we assume that train/valid dataloader are already multi-gpu aware
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.train_sampler = train_sampler
        self.valid_dataloader = valid_dataloader
        self.batch_transform_fn = batch_transform_fn
        self.get_metrics_fn = get_metrics_fn

        self.args = args

        self.per_worker_batch_size = self.args.batch_size * self.args.gradient_accumulation_steps
        self.global_batch_size = self.per_worker_batch_size * hvd.size()

        if self.args.clip_grad_norm is not None and self.args.clip_grad_value is not None:
            raise RuntimeError(f'Only one from clip_grad_norm and clip_grad_value should be set, but found '
                               f'clip_grad_norm = {self.args.clip_grad_norm}, '
                               f'clip_grad_value = {self.args.clip_grad_value}.')

        if self.args.clip_grad_norm or self.args.clip_grad_value:
            self.clip_grad = True
        else:
            self.clip_grad = False

        if hvd.rank() == 0:
            self.tb = SummaryWriter(log_dir=self.args.model_path)

        # move model to gpu
        self.model.cuda()

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if self.args.fp16_allreduce else hvd.Compression.none
        # Horovod: wrap optimizer with DistributedOptimizer.
        self.optimizer = hvd.DistributedOptimizer(self.optimizer,
                                                  named_parameters=self.model.named_parameters(),
                                                  compression=compression,
                                                  op=hvd.Average,
                                                  gradient_predivide_factor=1.0,
                                                  backward_passes_per_step=self.args.gradient_accumulation_steps,
                                                  )

        if args.lr_scheduler:
            if args.lr is None:
                raise RuntimeError('Set learning_rate to use learning rate schedulers.')
            if args.num_training_steps is None:
                args.num_training_steps = args.iters
            self.lr_scheduler = get_scheduler(args.lr_scheduler, self.optimizer,
                                              args.num_warmup_steps, args.num_training_steps)
        else:
            self.lr_scheduler = None

        # Apex
        if args.fp16:
            try:
                self.amp = importlib.import_module('apex.amp')
            except ImportError:
                raise ImportError('Install NVIDIA APEX to use fp16 training! Check README.md for instructions.')
            self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer,
                                                             enabled=self.args.fp16, opt_level=self.args.apex_opt_lvl,
                                                             min_loss_scale=self.args.min_loss_scale,
                                                             verbosity=int(hvd.rank() == 0))

        self.n_iter = 0
        self.n_epoch = 0
        if self.args.init_checkpoint:
            self.load(args.init_checkpoint)

    def step(self, batch, is_train_mode=True) -> Dict[str, float]:
        """Performs one step (forward and optionally backward and optimizer.step()) over data in a batch.

        Batch is splitted on sub-batches of self.args.batch_size size, loss and gradients are accumulated.

        Args:
            batch (dict): dict with inputs, inputs_mask, targets
            is_train_mode (bool, optional): In train mode we compute gradients, do backprop and optimizer.step().
                Defaults to True.

        Returns:
            float: loss on batch
        """
        batch_size = self.args.batch_size
        if is_train_mode:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        if self.batch_transform_fn:
            batch = self.batch_transform_fn(batch)
        for k in batch:
            batch[k] = batch[k].cuda()

        batch_metrics = defaultdict(lambda: 0.0)
        with torch.set_grad_enabled(is_train_mode):
            for j in range(0, len(batch['input_ids']), batch_size):
                subbatch = {k: batch[k][j: j + batch_size] for k in batch}
                outputs = self.model(**subbatch)
                loss = outputs['loss']
                metrics = self.get_metrics_fn(subbatch, outputs)

                # divide loss on gradient_accumulation_steps to get average loss for sub-batches
                loss = loss / self.args.gradient_accumulation_steps
                for k in metrics:
                    metrics[k] = metrics[k] / self.args.gradient_accumulation_steps
                    batch_metrics[k] += metrics[k].detach().item()

                if is_train_mode:
                    if self.args.fp16:
                        with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                            # last sub-batch, call synchronize within amp.scale_loss scope
                            # mb move to just above with optimizer.skip_synchronize()
                            if j == (len(batch['input_ids']) // batch_size - 1) * batch_size:
                                self.optimizer.synchronize()
                    else:
                        loss.backward()

            if is_train_mode:
                if self.args.fp16:
                    if self.clip_grad:
                        # grads already in sync
                        self._clip_gradients()
                    with self.optimizer.skip_synchronize():
                        self.optimizer.step()
                else:
                    if self.clip_grad:
                        self.optimizer.synchronize()
                        self._clip_gradients()
                        with self.optimizer.skip_synchronize():
                            self.optimizer.step()
                    else:
                        self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
        return batch_metrics

    def _clip_gradients(self):
        if self.args.fp16:
            # as recommended in https://nvidia.github.io/apex/advanced.html#gradient-clipping
            params = self.amp.master_params(self.optimizer)
        else:
            params = self.model.parameters()
        if self.args.clip_grad_value:
            torch.nn.utils.clip_grad_value_(params, self.args.clip_grad_value)
        elif self.args.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(params, self.args.clip_grad_norm)

    def _train_batch_generator(self):
        while self.n_iter <= self.args.iters:
            if self.train_sampler:
                self.train_sampler.set_epoch(self.n_epoch)
            # self.train_dataloader
            for batch in self.train_dataloader:
                if self.n_iter > self.args.iters:
                    return
                yield batch
                self.n_iter += 1
            self.n_epoch += 1

    def _skip_n_train_batches(self, train_batches, n):
        # we have to re-iterate over dataset
        # currently, skipping is based on number of iterations, not samples seen on previous run:
        #   (n_gpus x bs x n_grad_acc x n_iters)
        # todo: save number of seen samples in checkpoint
        if hvd.rank() == 0:
            logger.info(f'Skipping {n} batches from the dataset from epoch {self.n_epoch}...')
        # skipping...
        for _ in tqdm(itertools.islice(train_batches, n), disable=(hvd.rank() != 0), desc='Skipping...', total=n):
            ...

    def train(self) -> None:
        pbar = None
        if hvd.rank() == 0:
            pbar = tqdm(total=self.args.iters, desc='Train')
            pbar.update(self.n_iter)

        train_batches = self._train_batch_generator()

        # skip used data if needed
        if self.args.skip_used_data and self.n_iter > 0:
            train_size = None
            try:
                train_size = len(self.train_dataloader)
            except TypeError as e:
                if hvd.rank() == 0:
                    logger.info(f"Can't get train_dataloader length:\n{e}")
            # if we know train_size and number of epochs passed -> jump to this epoch and re-iterate over remainders
            skip_iter = self.n_iter % train_size if train_size else self.n_iter
            self.n_iter = (self.n_iter // train_size) * train_size if train_size else 0
            self._skip_n_train_batches(train_batches, skip_iter)

        metrics = defaultdict(lambda: [])
        best_valid_loss = np.inf
        valid_loss = np.nan
        train_loss = np.nan
        for batch in train_batches:
            iteration_start = time.time()
            batch_metrics = self.step(batch, is_train_mode=True)
            iteration_time = time.time() - iteration_start
            for k in batch_metrics:
                metrics[k] += [batch_metrics[k]]

            # logging
            if self.n_iter % self.args.log_interval == 0:
                # mean loss over last log_interval iterations
                metrics_mean = {}
                for k in metrics.keys():
                    metrics_mean[k] = list(itertools.chain.from_iterable(hvd.allgather_object(metrics[k])))
                    metrics_mean[k] = np.mean(metrics_mean[k])
                train_loss = metrics_mean['loss']
                metrics = defaultdict(lambda: [])
                if hvd.rank() == 0:
                    # todo: move logging, move to self.log()
                    for k in metrics_mean.keys():
                        logger.info(f'step: {self.n_iter}/{self.args.iters} {k}: {metrics_mean[k]:.4f}')
                        self.tb.add_scalar(f'{k}/iterations/train', metrics_mean[k], self.n_iter)
                        self.tb.add_scalar(f'{k}/samples/train', metrics_mean[k],
                                           self.n_iter * self.global_batch_size)
                    # log iteration time
                    self.tb.add_scalar('time/iterations/per_iter', iteration_time, self.n_iter)
                    self.tb.add_scalar('time/samples/per_iter', iteration_time,
                                       self.n_iter * self.global_batch_size)
                    # log learning rate
                    for j, param_group in enumerate(self.optimizer.param_groups):
                        # adafactor uses external lr to compute its own lr if scale_parameter is true
                        # adafactor might not have external lr in case if relative_step is used
                        for p in ['lr', 'scaled_lr']:
                            if p in param_group and param_group[p] is not None:
                                self.tb.add_scalar(f'{p}/iterations/param_group_{j}', param_group[p], self.n_iter)
                                self.tb.add_scalar(f'{p}/samples/param_group_{j}', param_group[p],
                                                   self.n_iter * self.global_batch_size)

            # validation
            if self.valid_dataloader is not None and self.n_iter % self.args.valid_interval == 0:
                # todo: we can use other metrics than loss here
                valid_metrics = self.validate(self.valid_dataloader)
                valid_loss = valid_metrics['loss']
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if self.args.save_best:
                        self.save(self.args.model_path, suffix='best', metrics=valid_metrics)

            # saving model
            if self.n_iter % self.args.save_interval == 0:
                self.save(self.args.model_path)

            if hvd.rank() == 0:
                pbar.update(1)
                pbar.set_postfix({'train_loss': f'{train_loss:.3f}',
                                  'valid_loss': f'{valid_loss:.3f}',
                                  'best_valid_loss': f'{best_valid_loss:.3f}'
                                  })

        if hvd.rank() == 0:
            # todo: run validation, call save model?
            pbar.close()
            logger.info('Done!')

    def validate(self, dataloader) -> Dict[str, float]:
        if hvd.rank() == 0:
            logger.info(f'start validation at step {self.n_iter}')

        metrics = defaultdict(lambda: [])
        for batch in tqdm(dataloader, desc='Validation', disable=(hvd.rank() != 0)):
            batch_metrics = self.step(batch, is_train_mode=False)
            for k in batch_metrics:
                metrics[k] += [batch_metrics[k]]

        metrics_mean = {}
        for k in metrics.keys():
            metrics_mean[k] = list(itertools.chain.from_iterable(hvd.allgather_object(metrics[k])))
            metrics_mean[k] = np.mean(metrics_mean[k])
        if hvd.rank() == 0:
            for k in metrics_mean.keys():
                logger.info(f'Validation {k}: {metrics_mean[k]:.4f}')
                self.tb.add_scalar(f'{k}/iterations/valid', metrics_mean[k], self.n_iter)
                self.tb.add_scalar(f'{k}/samples/valid', metrics_mean[k], self.n_iter * self.global_batch_size)
        return metrics_mean

    def load(self, load_path) -> None:
        # todo: if there is checkpoint in model_path load model from the latest checkpoint (init_checkpoint is None)
        checkpoint = torch.load(load_path, map_location='cpu')
        missing_k, unexpected_k = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if hvd.rank() == 0:
            if len(missing_k) != 0:
                logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
            if len(unexpected_k) != 0:
                logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler and not self.args.reset_lr:
            # if set reset_lr we do not load lr_scheduler and keep only the new one from __init__
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if 'amp' in checkpoint and self.args.fp16:
            self.amp.load_state_dict(checkpoint['amp'])
        self.n_iter = checkpoint.get('iteration', 0) + 1  # as saved iteration is already performed
        self.n_epoch = checkpoint.get('epoch', 0)
        if hvd.rank() == 0:
            logger.info(f'Model was loaded from: {self.args.init_checkpoint}')
            logger.info(f'Start iteration = {self.n_iter}')
            if self.lr_scheduler and self.args.reset_lr:
                logger.warning('lr_scheduler is not loaded from the checkpoint. New lr_scheduler is used with starting'
                               ' step (torch.optim.LRScheduler.__init__ last_epoch parameter) = -1.'
                               ' Current iteration number is ignored.')

    def save(self, save_path, suffix='', metrics=None) -> None:
        if hvd.rank() == 0:
            if suffix == '':
                save_path = f'{self.args.model_path}/model_{self.n_iter}.pth'
            else:
                save_path = f'{self.args.model_path}/model_{suffix}.pth'
            to_save = {
                       "model_state_dict": self.model.state_dict(),
                       "optimizer_state_dict": self.optimizer.state_dict(),
                       "iteration": self.n_iter,
                       "epoch": self.n_epoch,
                       }
            if metrics:
                to_save['metrics'] = metrics
            if self.args.fp16:
                to_save['amp'] = self.amp.state_dict()
            if self.lr_scheduler:
                to_save['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
            torch.save(to_save, save_path)
            logger.info(f'Model was saved to {save_path}')
