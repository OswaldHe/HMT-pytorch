import importlib
import itertools
import logging
import time

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
                 train_sampler=None, batch_transform_fn=None) -> None:
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

        self.args = args

        self.per_worker_batch_size = self.args.batch_size * self.args.gradient_accumulation_steps
        self.global_batch_size = self.per_worker_batch_size * hvd.size()

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
                                                             verbosity=int(hvd.rank() == 0))

        self.n_iter = 0
        self.n_epoch = 0
        if self.args.init_checkpoint:
            self.load(args.init_checkpoint)

    def step(self, batch, is_train_mode=True) -> float:
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

        batch_loss = 0
        with torch.set_grad_enabled(is_train_mode):
            for j in range(0, len(batch['input_ids']), batch_size):
                subbatch = {k: batch[k][j: j + batch_size] for k in batch}
                outputs = self.model(**subbatch)
                if self.args.fp16 and self.args.apex_opt_lvl in ['O2', 'O3']:
                    loss = outputs['loss']
                else:
                    loss = outputs.loss

                # divide loss on gradient_accumulation_steps to get average loss for sub-batches
                loss = loss / self.args.gradient_accumulation_steps
                batch_loss += loss.detach().item()

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
                    with self.optimizer.skip_synchronize():
                        self.optimizer.step()
                else:
                    self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
        return batch_loss

    def train(self) -> None:
        pbar = None
        if hvd.rank() == 0:
            pbar = tqdm(total=self.args.iters, desc='Train')
            pbar.update(self.n_iter)

        if self.n_iter > 0:
            self._skip_n_train_batches(self.n_iter - 1)

        losses = []
        best_valid_loss = np.inf
        valid_loss = np.nan
        train_loss = np.nan
        while self.n_iter <= self.args.iters:
            if self.train_sampler:
                # to shuffle data in each epoch differently
                self.train_sampler.set_epoch(self.n_epoch)
            for batch in self.train_dataloader:
                if self.n_iter > self.args.iters:
                    return self._stop_training(pbar)
                iteration_start = time.time()
                batch_loss = self.step(batch, is_train_mode=True)
                iteration_time = time.time() - iteration_start
                losses += [batch_loss]

                # logging
                if self.n_iter % self.args.log_interval == 0:
                    losses = list(itertools.chain.from_iterable(hvd.allgather_object(losses)))
                    # mean loss over last log_interval iterations
                    train_loss = np.mean(losses)
                    losses = []
                    if hvd.rank() == 0:
                        # todo: move logging, move to self.log()
                        logger.info(f'step: {self.n_iter}/{self.args.iters} loss: {train_loss:.4f}')
                        self.tb.add_scalar('loss/iterations/train', train_loss, self.n_iter)
                        self.tb.add_scalar('loss/samples/train', train_loss, self.n_iter * self.global_batch_size)
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
                    valid_loss = self.validate(self.valid_dataloader)
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        if self.args.save_best:
                            self.save(self.args.model_path, suffix='best')

                # saving model
                if self.n_iter % self.args.save_interval == 0:
                    self.save(self.args.model_path)

                self.n_iter += 1
                if hvd.rank() == 0:
                    pbar.update(1)
                    pbar.set_postfix({'train_loss': f'{train_loss:.3f}',
                                      'valid_loss': f'{valid_loss:.3f}',
                                      'best_valid_loss': f'{best_valid_loss:.3f}'
                                      })
            self.n_epoch += 1

        self._stop_training(pbar)

    def _stop_training(self, pbar):
        # todo: run validation, call save model?
        if hvd.rank() == 0:
            pbar.close()
            logger.info('Done!')

    def validate(self, dataloader) -> float:
        if hvd.rank() == 0:
            logger.info(f'start validation at step {self.n_iter}')

        losses = []
        for batch in tqdm(dataloader, desc='Validation', disable=(hvd.rank() != 0)):
            batch_loss = self.step(batch, is_train_mode=False)
            losses += [batch_loss]

        losses = list(itertools.chain.from_iterable(hvd.allgather_object(losses)))
        mean_loss = np.mean(losses)
        # logging
        if hvd.rank() == 0:
            logger.info(f'valid_loss: {mean_loss:.4f}')
            self.tb.add_scalar('loss/iterations/valid', mean_loss, self.n_iter)
            self.tb.add_scalar('loss/samples/valid', mean_loss, self.n_iter * self.global_batch_size)
        return mean_loss

    def _skip_n_train_batches(self, n):
        # todo: we can skip directly to n_epoch
        pbar = None
        if hvd.rank() == 0:
            logger.info(f'Skipping first {n} batches from the dataset...')
            pbar = tqdm(total=n, desc='Skipping...')

        i = 0
        epoch = 0
        while i < n:
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            for _ in self.train_dataloader:
                if i >= n:
                    break
                i += 1
                if hvd.rank() == 0:
                    pbar.update(1)
            epoch += 1
        if hvd.rank() == 0:
            pbar.close()

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
                logger.warning(f'lr_scheduler is not loaded from the checkpoint. New lr_scheduler is used with starting'
                               f' step (torch.optim.LRScheduler last_epoch parameter) = {self.n_iter}')

    def save(self, save_path, suffix='') -> None:
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
            if self.args.fp16:
                to_save['amp'] = self.amp.state_dict()
            if self.lr_scheduler:
                to_save['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
            torch.save(to_save, save_path)
            logger.info(f'Model was saved to {save_path}')
