# t5-experiments

## QQP
QQP is currently not available via tfds: https://github.com/tensorflow/datasets/pull/3031

to hot-fix this go to the source code of installed tfds `tensorflow_datasets/text/glue.py:215` and replace data url with https://dl.fbaipublicfiles.com/glue/data/QQP.zip