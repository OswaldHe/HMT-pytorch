import datetime
import json
import time
from typing import Optional, Iterable, Tuple, Collection, Any, List
from itertools import islice, chain


from deeppavlov.core.trainers import TorchTrainer
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.trainers.utils import NumpyArrayEncoder, prettify_metrics, Metric

import horovod.torch as hvd


import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
hvd.init()


def loggger_info(*args):
    if hvd.rank() == 0:
        logger.info(*args)


class HvdTorchNNTrainer(TorchTrainer):
    """DeepPavlov Horovod trainer.

    Is working, but does not support:
        - safe-exit with evaluation on ctrl+c

    TODO:
        - check that all necessary stats are synchronized (batches seen and etc)
        - remove debug logging

    python -m deeppavlov train/evaluate with Horovod was tested
    the same config w & w/o horovodrun produces the same results (python -m evaluation)

    e.g. commands:
    with horovod
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; horovodrun --gloo -np 8 \
            python -m deeppavlov evaluate ./dp_configs/wmt/ende_hvd.json
    w/o:
        export CUDA_VISIBLE_DEVICES=0; python -m deeppavlov evaluate ./dp_configs/wmt/ende_hvd.json
    """
    def __init__(self, *args, **kwargs):
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'hvd rank: {hvd.rank()}')
        super().__init__(*args, **kwargs)

    def save(self) -> None:
        if hvd.rank() == 0:
            logger.info('HvdTorchNNTrainer.save()')
            super(TorchTrainer, self).save()

    def _log(self, iterator: DataLearningIterator,
             tensorboard_tag: Optional[str] = None, tensorboard_index: Optional[int] = None) -> None:
        self._send_event(event_name='before_log')
        if self.log_on_k_batches == 0:
            report = {
                'time_spent': str(datetime.timedelta(seconds=round(time.time() - self.start_time + 0.5)))
            }
        else:
            data = islice(iterator.gen_batches(self.batch_size, data_type='train', shuffle=True),
                          self.log_on_k_batches)
            report = self.test(data, self.train_metrics, start_time=self.start_time)

        report.update({
            'epochs_done': self.epoch,
            'batches_seen': self.train_batches_seen,
            'train_examples_seen': self.examples
        })

        metrics: List[Tuple[str, float]] = list(report.get('metrics', {}).items()) + list(self.last_result.items())

        report.update(self.last_result)

        if self.losses:
            self.losses = list(chain.from_iterable(hvd.allgather_object(self.losses)))
            report['loss'] = sum(self.losses) / len(self.losses)
            self.losses.clear()
            metrics.append(('loss', report['loss']))

        # todo: we might need to gather other stats from report (metrics already gatherer)
        # for train_examples_seenwe can simpy multilpy by hvd.size()
        report['train_examples_seen'] = report.get('train_examples_seen', 0) * hvd.size()

        # writing to tensorboard should be done by one worker
        if hvd.rank() == 0:
            if metrics and self.tensorboard_log_dir is not None:
                with self.tb_train_writer.as_default() as writer:
                    for name, score in metrics:
                        self._tf.summary.scalar(name=f'{tensorboard_tag}/{name}', data=score, step=tensorboard_index)
                    writer.flush()

        self._send_event(event_name='after_train_log', data=report)

        report = {'train': report}
        if hvd.rank() == 0:
            print(json.dumps(report, ensure_ascii=False, cls=NumpyArrayEncoder))

    def _validate(self, iterator: DataLearningIterator,
                  tensorboard_tag: Optional[str] = None, tensorboard_index: Optional[int] = None) -> None:
        logger.info('HvdTorchNNTrainer._validate()')
        self._send_event(event_name='before_validation')
        # report from self.test is already gathered from all workers
        report = self.test(iterator.gen_batches(self.batch_size, data_type='valid', shuffle=False),
                           start_time=self.start_time)

        report['epochs_done'] = self.epoch
        report['batches_seen'] = self.train_batches_seen
        report['train_examples_seen'] = self.examples

        metrics = list(report['metrics'].items())

        # write to tensorboard only from one worker
        if hvd.rank() == 0:
            if tensorboard_tag is not None and self.tensorboard_log_dir is not None:
                with self.tb_valid_writer.as_default() as writer:
                    for name, score in metrics:
                        self._tf.summary.scalar(name=f'{tensorboard_tag}/{name}', data=score, step=tensorboard_index)
                    writer.flush()

        m_name, score = metrics[0]

        # Update the patience
        if self.score_best is None:
            self.patience = 0
        else:
            if self.improved(score, self.score_best):
                self.patience = 0
            else:
                self.patience += 1

        # Run the validation model-saving logic
        if self._is_initial_validation():
            loggger_info('Initial best {} of {}'.format(m_name, score))
            self.score_best = score
        elif self._is_first_validation() and self.score_best is None:
            loggger_info('First best {} of {}'.format(m_name, score))
            self.score_best = score
            loggger_info('Saving model')
            self.save()
        elif self.improved(score, self.score_best):
            loggger_info('Improved best {} of {}'.format(m_name, score))
            self.score_best = score
            loggger_info('Saving model')
            self.save()
        else:
            loggger_info('Did not improve on the {} of {}'.format(m_name, self.score_best))

        report['impatience'] = self.patience
        if self.validation_patience > 0:
            report['patience_limit'] = self.validation_patience

        self._send_event(event_name='after_validation', data=report)
        report = {'valid': report}
        if hvd.rank() == 0:
            print(json.dumps(report, ensure_ascii=False, cls=NumpyArrayEncoder))
        self.validation_number += 1

    def test(self, data: Iterable[Tuple[Collection[Any], Collection[Any]]],
             metrics: Optional[Collection[Metric]] = None, *,
             start_time: Optional[float] = None, show_examples: Optional[bool] = None) -> dict:
        """
        Calculate metrics and return reports on provided data for currently stored
        :class:`~deeppavlov.core.common.chainer.Chainer`

        Args:
            data: iterable of batches of inputs and expected outputs
            metrics: collection of metrics namedtuples containing names for report, metric functions
                and their inputs names (if omitted, ``self.metrics`` is used)
            start_time: start time for test report
            show_examples: a flag used to return inputs, expected outputs and predicted outputs for the last batch
                in a result report (if omitted, ``self.show_examples`` is used)

        Returns:
            a report dict containing calculated metrics, spent time value, examples count in tested data
            and maybe examples
        """
        logger.info('HvdTorchNNTrainer.test()')
        self._chainer.get_main_component().model.eval()
        if start_time is None:
            start_time = time.time()
        if show_examples is None:
            show_examples = self.show_examples
        if metrics is None:
            metrics = self.metrics

        expected_outputs = list(set().union(self._chainer.out_params, *[m.inputs for m in metrics]))

        outputs = {out: [] for out in expected_outputs}
        examples = 0

        data = islice(data, self.max_test_batches)

        for x, y_true in data:
            examples += len(x)
            y_predicted = list(self._chainer.compute(list(x), list(y_true), targets=expected_outputs))
            if len(expected_outputs) == 1:
                y_predicted = [y_predicted]
            for out, val in zip(outputs.values(), y_predicted):
                out += list(val)

        if examples == 0:
            logger.warning('Got empty data iterable for scoring')
            return {'eval_examples_count': 0, 'metrics': None, 'time_spent': str(datetime.timedelta(seconds=0))}

        # metrics_values = [(m.name, m.fn(*[outputs[i] for i in m.inputs])) for m in metrics]

        # for k, v in outputs.items():
        #     logger.info(f'{k}: {v}')

        # gather data from all workers
        # logger.info('gathering outputs')
        for k in sorted(outputs.keys()):
            outputs[k] = list(chain.from_iterable(hvd.allgather_object(outputs[k])))
            # logger.info(f'{k}: {outputs[k]}')

        # for k, v in outputs.items():
        #     logger.info(f'{k}: {len(v)}')

        examples = sum(hvd.allgather_object(examples))

        metrics_values = []
        for metric in metrics:
            value = metric.fn(*[outputs[i] for i in metric.inputs])
            metrics_values.append((metric.alias, value))

        report = {
            'eval_examples_count': examples,
            'metrics': prettify_metrics(metrics_values),
            'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
        }

        if show_examples:
            y_predicted = zip(*[y_predicted_group
                                for out_name, y_predicted_group in zip(expected_outputs, y_predicted)
                                if out_name in self._chainer.out_params])
            if len(self._chainer.out_params) == 1:
                y_predicted = [y_predicted_item[0] for y_predicted_item in y_predicted]
            report['examples'] = [{
                'x': x_item,
                'y_predicted': y_predicted_item,
                'y_true': y_true_item
            } for x_item, y_predicted_item, y_true_item in zip(x, y_predicted, y_true)]

        self._chainer.get_main_component().model.train()
        return report
