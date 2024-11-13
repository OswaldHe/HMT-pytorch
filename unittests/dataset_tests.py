import unittest
from hmt_tools.data_processing.narrativeqa import load_narrativeqa_test, load_narrativeqa_train_valid
from datasets import Dataset

class TestLoadQADataset(unittest.TestCase):

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    def test_narrativeqa_test(self):
        ds_test = load_narrativeqa_test(max_token_num=12800, test_length=10000, block_size=1024, tokenizer=self.tokenizer)
        
        self.assertIsInstance(ds_test, Dataset)
        self.assertEqual(set(ds_test.column_names), {'text', 'labels', 'labels_length'})
        
        print("First entry of single split dataset:")
        print(ds_test[0])
        print("Mask size of the first entry:")
        print(ds_test[0]['labels_length'])

    def test_narrativeqa_train_valid(self):
        ds_train, ds_valid = load_narrativeqa_train_valid(max_token_num=12800, block_size=1024, tokenizer=self.tokenizer, split=['train[:5%]', 'validation[:5%]'])
        
        self.assertIsInstance(ds_train, Dataset)
        self.assertIsInstance(ds_valid, Dataset)
        
        for ds in [ds_train, ds_valid]:
            self.assertEqual(set(ds.column_names), {'text', 'labels', 'labels_length'})
        
        print("First entry of train split:")
        print(ds_train[0])
        print("First entry of validation split:")
        print(ds_valid[0])

    def test_qmsum_train(self):
        from hmt_tools.data_processing.qmsum import load_qmsum_train

        ds = load_qmsum_train(max_token_num=12000, block_size=1024, tokenizer=self.tokenizer, path="./repo/QMSum/data/train.jsonl")
        print("QMSum train dataset:")
        print(ds)
        
        print("\nColumns in the dataset:")
        print(ds.column_names)

        print("\nKeys of the first datapoint:")
        print(ds[0].keys())
        

        self.assertIsInstance(ds, Dataset)
        self.assertTrue('input_ids' in ds.column_names)
        self.assertTrue('mask_size' in ds.column_names)

        # Check if the dataset is not empty
        self.assertGreater(len(ds), 0)
    
    
    def test_qmsum_train_huggingface(self):
        from hmt_tools.data_processing.qmsum import load_qmsum_train

        total_ds = load_qmsum_train(max_token_num=12000, block_size=1024, tokenizer=self.tokenizer, source='huggingface')
        splited_dict = total_ds.train_test_split(test_size=0.2)
        train_ds = splited_dict['train']
        valid_ds = splited_dict['test']

        print("QMSum train dataset:")
        print(train_ds)
        
        print("QMSum valid dataset:")
        print(valid_ds)

    def test_musique_test(self):
        from hmt_tools.data_processing.prep_funcs import prepare_musique_test_ppl
        from hmt_tools.data_processing.generic import prepare_test
        from datasets import load_dataset
        ds = load_dataset(path="THUDM/LongBench", name="musique", split='test', streaming=False, trust_remote_code=True)
        ds = prepare_test(ds, prepare_musique_test_ppl, max_token_num=12800, test_length=10000, block_size=1024, tokenizer=self.tokenizer, with_answer=True)

        print("MuSiQue test dataset:")
        print(ds)

        print("\nColumns in the dataset:")
        print(ds.column_names)

        print("\nKeys of the first datapoint:")
        print(ds[0].keys())

        print("\nFirst three data points:")
        for i in range(3):
            print(f"\nData point {i+1}:")
            print("input_ids: ", ds['input_ids'][i][:100], "...")  # Print first 100 characters
            print("mask_size: ", ds['mask_size'][i])

    def test_musique_train(self):
        from hmt_tools.data_processing.musique import load_musique_train

        ds = load_musique_train(max_token_num=12000, block_size=1024, tokenizer=self.tokenizer)
        print("MuSiQue train dataset:")
        print(ds)
            
        print("\nColumns in the dataset:")
        print(ds.column_names)

        print("\nKeys of the first datapoint:")
        print(ds[0].keys())

        print("\nFirst three data points:")
        for i in range(3):
            print(f"\nData point {i+1}:")
            print("input_ids: ", ds['input_ids'][i][:100], "...")
            print("mask_size: ", ds['mask_size'][i])


class TestLoadRedPajama(unittest.TestCase):
    def test_load_redpajama_test(self):
        """Trying to load the redpajama test split, which is the last 10% of the dataset. Also testing dataloader creation from the dataset.
        """
        from hmt_tools.data_processing.red_pajamav2 import load_redpajama
        test_ds = load_redpajama(tokenizer=self.tokenizer, split='train[90%:]', history_size=10000, block_size=990, streaming=False, trust_remote_code=True)

        print(type(test_ds))
        print(test_ds)
        print(len(test_ds))

        print(test_ds[0])

        batch_size = 2
        from torch.utils.data import DataLoader
        from hmt_tools.collate import collate_fn
        from functools import partial
        collate_fn = partial(collate_fn, id_pad_value=self.tokenizer.pad_token_id, is_qa_task=False, block_size=990, batch_size=batch_size)
        dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        print(len(dataloader))
        data_iter = iter(dataloader)
        print(next(data_iter))


if __name__ == '__main__':
    unittest.main()
