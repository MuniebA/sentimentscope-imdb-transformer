# SentimentScope: IMDB sentiment analysis with transformers

Binary sentiment classification of IMDB movie reviews (Udacity AWS AI ML Scholars project).

| Model | Test accuracy | File |
|---|---|---|
| DemoGPT, transformer trained from scratch (required part) | 81.53% | `sentimentscope_best.pt` (in this repo) |
| Fine-tuned `bert-base-uncased` (extension) | 92.20% | `sentimentscope_bert_best.pt` ([download from the Release](https://github.com/MuniebA/sentimentscope-imdb-transformer/releases/download/v1.0/sentimentscope_bert_best.pt), ~220 MB, half precision) |

- `SentimentScope_starter.ipynb` contains the full workflow: data exploration, DemoGPT, training, evaluation, inference classes and the conclusion.
- The BERT checkpoint is stored in half precision to reduce its size; it is the same model that scored 92.20% on the full-precision weights.
- Inference: `SentimentPredictor` and `BertSentimentPredictor` in the notebook load a checkpoint and classify a batch of raw review text.
