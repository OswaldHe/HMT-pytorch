# ECtHR-NAACL2021

link: https://archive.org/details/ECtHR-NAACL2021

download & unzip data: https://archive.org/download/ECtHR-NAACL2021/dataset.zip

## ECTHRBinaryClassification
Binary classification version of the dataset: based on list of facts predict if any article was violated.
For balanced binary classification set `subsample='balanced'`.

| model            | n_segm | input_seq_len | accuracy (test) | accuracy (valid) | bs  | lr    | optimizer | scheduler | wd    | steps | patience |
| ---------------- | ------ | ------------- | --------------- | ---------------- | --- | ----- | --------- | --------- | ----- | ----- | -------- |
| RMT roberta-base | 1      | 499           | 71.60 +- 2.60   | 73.65 +- 0.89    | 32  | 1e-05 | AdamW     | constant  | 1e-03 | 3200  | 15       |
| RMT roberta-base | 2      | 998           | 71.48 +- 0.64   | 74.66 +- 0.89    | 32  | 1e-05 | AdamW     | constant  | 1e-03 | 3200  | 15       |
| RMT roberta-base | 4      | 1996          | 72.10 +- 1.13   | 76.01 +- 1.47    | 32  | 1e-05 | AdamW     | constant  | 1e-03 | 3200  | 15       |
| RMT roberta-base | 8      | 3992          | 72.96 +- 1.61   | 74.55 +- 0.39    | 32  | 2e-05 | AdamW     | linear    | 1e-03 | 3200  | 15       |
| RMT roberta-base | 16     | 7984          | 72.74 +- 1.99   | 75.47 +- 0.88   | 32  | 1e-05 | AdamW     | constant    | 1e-03 | 3200  | 15       |