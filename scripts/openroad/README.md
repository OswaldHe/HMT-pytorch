### Instructions
1. Clone the repository of RAG-EDA: https://github.com/OswaldHe/RAG-EDA.
2. Pass the file paths to `OpenROAD` and `OpenROAD_test` functions.
3. Start the training/evaluation script in this folder.
4. Link to finetuned model: https://huggingface.co/OswaldHe123/HMT-Llama3.1-8B-OpenROAD
5. Please download the model locally and use `from_pretrained` to load the model after the instantiation of `RecurrentWrapper`.
6. Move `memory.pt` to `pwd` to use pre-computed memory embedding for fast generation (~0.1s response time for each sample).