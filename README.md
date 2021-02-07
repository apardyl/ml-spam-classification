## How to begin:

1. Clone this repository and enter it.
2. `python -m venv venv`
3. `source venv/bin/activate`
4. `pip install -r requirements.txt`

## How to train models:

1. Place your favourite datasets in the `data` directory with the following layout:
    - Directory layout:
        - `data/`
            - `dataset_name/`
                - `ham/` # place your ham/clean message samples here.
                - `spam/` # place spam samples here.
    - Messages can be divided into multiple subdirectories (e.g. `data/dataset_name/spam/abc/1234/message1)
    - All messages must be RAW emails (with headers and proper encoding). A Maildir-like mailbox items should work out
      of the box.
    - Recommended datasets:
        - Enron RAW dataset: http://www2.aueb.gr/users/ion/data/enron-spam/
        - Spamassassin public corpus dataset: https://spamassassin.apache.org/old/publiccorpus/
2. Run `python datasets.py --dataset dataset_name --language language_code` to preprocess the datasets (this will take
   some time).
    - Multiple warnings are expected - this is due to the facts, that spam messages are often malformed or contain only
      images and not text.
    - Supported language values: `en`, `pl`
4. Run `python train.py --dataset dataset_name` to train the classifier (this will take a lot of time, you can stop and
   resume the process at any time). Model will be saved in `saved_models/`.
5. (optional) review training results with Tensorboard: `tensorboard --logdir runs` (can also be done during training).

## How to use the classifier:

1. Run `python classifier.py --classifier saved_model_path --message_dir message_dir`. The `saved_model_path` parameter
   should point to a saved model state generated by `train.py`, `message_dir` is a path to a directory containing RAW
   email messages. Results will be printed to stdout. 

