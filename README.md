# MAC: Multi-faceted Aspect Correlation Mining for Review-based Recommendation
This is our PyTorch implementation for the paper.


## Requirements
- Python 3.8 
- Pytorch 1.12.1
- NVIDIA GPU + CUDA + CuDNN

## Running the code

Run word2vector.py for word embedding. Glove pretraining weight is required.
Make sure can run load_sentiment_data in load_data.py
Run BERT/bert_whitening.py for obtaining the feature vector for each review.
If previous steps successfully run, then you can run MAC.py.
