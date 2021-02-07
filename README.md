## How to begin:

1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`

## How to train models:

1. Download the Enron RAW dataset from here: http://www2.aueb.gr/users/ion/data/enron-spam/ and unpack all ham messages
   in data/enron/ham and all spam messages in data/enron/ham.
2. Download the Spamassassin public corpus dataset from here: https://spamassassin.apache.org/old/publiccorpus/ and
   unpack all ham messages in data/spamassassin/ham and all spam messages in data/spamassassin/ham.
3. Run `python datasets.py` to preprocess the datasets (this will take some time).
    - Multiple warnings are expected - this is due to the facts, that spam messages are often malformed or contain only
      images and not text.
4. Run `python train.py` to train the classifier (this will take a lot of time, you can stop and resume the process at any time).

## How to use the classifier:

