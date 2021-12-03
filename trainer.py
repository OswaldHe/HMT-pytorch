import importlib
import itertools
import logging
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import horovod.torch as hvd


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, optimizer, train_dataloader, valid_dataloader, args) -> None:
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
        self.valid_dataloader = valid_dataloader

        self.args = args

        self.per_worker_batch_size = self.args.batch_size * self.args.gradient_accumulation_steps
        self.global_batch_size = self.per_worker_batch_size * hvd.size()

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

        # Apex
        if args.fp16:
            try:
                self.amp = importlib.import_module('apex.amp')
            except ImportError:
                raise ImportError('Install NVIDIA APEX to use fp16 training! Check README.md for instructions.')
            self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer,
                                                             enabled=self.args.fp16, opt_level=self.args.apex_opt_lvl)

        self.n_iter = 0
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

        for k in batch:
            batch[k] = batch[k].cuda()

        batch_loss = 0
        with torch.set_grad_enabled(is_train_mode):
            for j in range(0, len(batch['inputs']), batch_size):
                outputs = self.model(input_ids=batch['inputs'][j: j + batch_size],
                                     attention_mask=batch['inputs_mask'][j: j + batch_size],
                                     # todo: use decoder_attention mask!
                                     # it is okay to not use decoder_attention_mask as loss from paddings is ignored
                                     # and decoder_attention_mask should be zero only for paddings
                                     # so, anyway padding are ignored.
                                     # decoder_attention_mask=batch['targets_mask'][j: j + args.batch_size],
                                     labels=batch['targets'][j: j + batch_size])
                if self.args.fp16 and self.args.apex_opt_lvl == 'O2':
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
                            if j == (len(batch['inputs']) // batch_size - 1) * batch_size:
                                self.optimizer.synchronize()
                    else:
                        loss.backward()

            if is_train_mode:
                if self.args.fp16:
                    with self.optimizer.skip_synchronize():
                        self.optimizer.step()
                else:
                    self.optimizer.step()
        return batch_loss

    def train(self) -> None:
        pbar = None
        if hvd.rank() == 0:
            pbar = tqdm(total=self.args.iters, desc='Train')
            pbar.update(self.n_iter)

        losses = []
        best_valid_loss = np.inf
        valid_loss = np.nan
        while self.n_iter <= self.args.iters:
            for batch in self.train_dataloader:
                if self.n_iter > self.args.iters:
                    break
                iteration_start = time.time()
                batch_loss = self.step(batch, is_train_mode=True)
                iteration_time = time.time() - iteration_start
                losses += [batch_loss]

                # logging
                if self.n_iter % self.args.log_interval == 0:
                    losses = list(itertools.chain.from_iterable(hvd.allgather_object(losses)))
                    mean_loss = np.mean(losses)
                    if hvd.rank() == 0:
                        # todo: move logging, move to self.log()
                        logger.info(f'step: {self.n_iter}/{self.args.iters} loss: {mean_loss:.4f}')
                        pbar.set_postfix({
                            'train_loss': f'{mean_loss:.3f}',
                            'valid_loss': f'{valid_loss:.3f}',
                            'best_valid_loss': f'{best_valid_loss:.3f}'
                            })
                        self.tb.add_scalar('loss/train', mean_loss, self.n_iter)
                        self.tb.add_scalar('loss/iterations/train', mean_loss, self.n_iter)
                        self.tb.add_scalar('loss/samples/train', mean_loss, self.n_iter * self.global_batch_size)
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
                    losses = []

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
            self.tb.add_scalar('loss/valid', mean_loss, self.n_iter)
            self.tb.add_scalar('loss/iterations/valid', mean_loss, self.n_iter)
            self.tb.add_scalar('loss/samples/valid', mean_loss, self.n_iter * self.global_batch_size)
        return mean_loss

    def load(self, load_path) -> None:
        # todo: use iteration number to restore position in dataset?
        # todo: if there is checkpoint in model_path load model from the latest checkpoint (init_checkpoint is None)
        checkpoint = torch.load(load_path, map_location='cpu')
        missing_k, unexpected_k = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if hvd.rank() == 0:
            if len(missing_k) != 0:
                logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
            if len(unexpected_k) != 0:
                logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if 'amp' in checkpoint and self.args.fp16:
            self.amp.load_state_dict(checkpoint['amp'])
        self.n_iter = checkpoint.get('iteration', 0)
        if hvd.rank() == 0:
            logger.info(f'Model was loaded from: {self.args.init_checkpoint}')
            logger.info(f'Start iteration = {self.n_iter}')

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
                       }
            if self.args.fp16:
                to_save['amp'] = self.amp.state_dict()
            torch.save(to_save, save_path)
            logger.info(f'Model was saved to {save_path}')
