import unittest
from tools.data_processing.hmt_qa_datasets import load_qa_dataset
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

if __name__ == '__main__':
    unittest.main()
