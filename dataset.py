# Put the Dataset class in a separate file so other files can unpickle it as well.
# This is not possible when a dataset is created in a __main__ module of a script

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


# THE DATASET CLASS
class Dataset:
    """
    Initializes the dataset class and creates the minibatches
    """
    def __init__(self, train_set, test_set, scaler, val_frac=0.05, shuffle=True,
                 max_name_len=10, max_desc_len=75, text_voc_size=501,
                 brand_voc_size=501, category_voc_size=201):
        train_set, valid_set = train_test_split(train_set, train_size=1 - val_frac)

        self.train_x = self.collect_features(train_set, max_name_len, max_desc_len)
        self.valid_x = self.collect_features(valid_set, max_name_len, max_desc_len)
        self.test_x = self.collect_features(test_set, max_name_len, max_desc_len)

        self.train_y = train_set['target'].as_matrix()
        self.valid_y = valid_set['target'].as_matrix()

        self.valid_prices = valid_set['price'].as_matrix()

        self.test_ids = test_set[['test_id']]

        self.scaler = scaler
        self.shuffle = shuffle
        self.text_voc_size = text_voc_size
        self.brand_voc_size = brand_voc_size
        self.category_voc_size = category_voc_size

    @staticmethod
    def collect_features(dataset, max_name_len, max_desc_len):
        features = {
            'name': pad_sequences(dataset.seq_name, maxlen=max_name_len),
            'item_desc': pad_sequences(dataset.seq_item_description, maxlen=max_desc_len),
            'brand_name': dataset.brand_name.as_matrix(),
            'category_name': dataset.category_name.as_matrix(),
            'item_condition': dataset.item_condition_id.as_matrix(),
            'shipping': dataset[['shipping']].as_matrix()
        }
        return features

    def batches(self, batch_size):
        if self.shuffle:
            idx = np.arange(len(self.train_x))
            np.random.shuffle(idx)
            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]

        # n_batches = len(self.train_y) // batch_size

        for ii in range(0, len(self.train_y), batch_size):
            x = self.train_x[ii:ii + batch_size]
            y = self.train_y[ii:ii + batch_size]

            yield x, y
