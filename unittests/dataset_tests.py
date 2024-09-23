import unittest
from tools.data_processing.hmt_qa_datasets import load_qa_dataset
from tools.data_processing.qmsum import load_qmsum_test
from datasets import Dataset

class TestLoadQADataset(unittest.TestCase):
    def test_narrativeqa_single_split(self):
        ds_test = load_qa_dataset(name="deepmind/narrativeqa", source="deepmind", split='test')
        
        self.assertIsInstance(ds_test, Dataset)
        self.assertEqual(set(ds_test.column_names), {'text', 'labels', 'labels_length'})
        
        print("First entry of single split dataset:")
        print(ds_test[0])
        print("Mask size of the first entry:")
        print(ds_test[0]['labels_length'])

    def test_narrativeqa_multiple_splits(self):
        ds_train, ds_valid, ds_test = load_qa_dataset(name="deepmind/narrativeqa", source="deepmind", split=['train[:1%]', 'train[1%:2%]', 'train[2%:3%]'])
        
        self.assertIsInstance(ds_train, Dataset)
        self.assertIsInstance(ds_valid, Dataset)
        self.assertIsInstance(ds_test, Dataset)
        
        for ds in [ds_train, ds_valid, ds_test]:
            self.assertEqual(set(ds.column_names), {'text', 'labels', 'labels_length'})
        
        print("First entry of train split:")
        print(ds_train[0])
        print("First entry of validation split:")
        print(ds_valid[0])
        print("First entry of test split:")
        print(ds_test[0])

    def test_qmsum_test(self):
        ds = load_qmsum_test(max_token_num=12800, test_length=10000, block_size=1024)
        print("QMSum test dataset:")
        print(ds)
        
        print("\nColumns in the dataset:")
        print(ds.column_names)

        print("\nKeys of the first datapoint:")
        print(ds[0].keys())
        
        
        print("\nFirst three data points:")
        for i in range(3):
            print(f"\nData point {i+1}:")
            print("Answer Length: ", ds[i]['answer_length'])
            print("Length of input_ids: ", len(ds[i]['input_ids']))
            print("Length of attention_mask: ", len(ds[i]['attention_mask']))
            print("Length of labels", len(ds[i]['labels']))
            print("mask_size", ds[i]['mask_size'])

        self.assertIsInstance(ds, Dataset)
        self.assertTrue('labels' in ds.column_names)
        self.assertTrue('input_ids' in ds.column_names)
        self.assertTrue('mask_size' in ds.column_names)

    def test_qmsum_train(self):
        from tools.data_processing.qmsum import load_qmsum_train

        ds = load_qmsum_train("/home/yingqi/repo/QMSum/data/train.jsonl")
        print("QMSum train dataset:")
        print(ds)
        
        print("\nColumns in the dataset:")
        print(ds.column_names)

        print("\nKeys of the first datapoint:")
        print(ds[0].keys())
        
        print("\nFirst three data points:")
        for i in range(3):
            print(f"\nData point {i+1}:")
            print("Text: ", ds['text'][i][:100] + "...")  # Print first 100 characters
            print("Answer Length: ", str(ds['answer_length'][i]), "...")  # Print first 100 characters

        self.assertIsInstance(ds, Dataset)
        self.assertTrue('text' in ds.column_names)
        self.assertTrue('answer_length' in ds.column_names)

        # Check if the dataset is not empty
        self.assertGreater(len(ds), 0)

        # Check if all entries have the expected fields
        for item in ds:
            self.assertTrue('text' in item)
            self.assertTrue('answer_length' in item)

        # Check if the lengths of tasks, meeting notes, and answers are consistent
        self.assertEqual(len(ds['text']), len(ds['answer_length']))

if __name__ == '__main__':
    unittest.main()
