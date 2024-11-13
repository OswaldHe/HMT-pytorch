from matplotlib import pyplot as plt

def dataset_length_histogram(dataset, split):
    """WARNINGL This shows the length of the text as a string, rather than number of tokens.

    :param dataset: Hugging Face dataset
    :type dataset: Dataset
    :param split: Name of the split to visualize, e.g. 'train' or 'validation'
    :type split: str
    """

    # Get a list of the lengths of the text in the validation dataset
    text_lengths = [len(sample['text']) for sample in dataset]

    print(f"Number of samples in {split} dataset: {len(text_lengths)}")
    print(f"First 10 text lengths: {text_lengths[:10]}")

    # Plot a histogram of the lengths to visualize the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=50)
    plt.title(f'Distribution of Text Lengths in {split} Dataset')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.show()

    # Plot the first 100 lengths
    plt.figure(figsize=(10, 6))
    plt.plot(text_lengths[:100])
    plt.title(f'First 100 Text Lengths in {split} Dataset')
    plt.xlabel('Sample Index')
    plt.ylabel('Text Length')
    plt.show()