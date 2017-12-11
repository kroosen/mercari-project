# Notebook for the Mercari Kaggle competition
# Write some stuff here
#

# General modules
import numpy as np
import math
import pickle
import os.path

# Modules for handling data
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from dataset import Dataset

from keras.preprocessing.text import Tokenizer

# Modules for building the neural network
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k


# THE GIVEN SCORING FUNCTION
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, _ in enumerate(y_pred)]

    return math.sqrt(sum(to_sum) * (1.0 / len(y)))
    # https://www.kaggle.com/c/mercari-price-suggestion-challenge#evaluation


# LOADING THE DATA
def load_data():
    train_set = pd.read_table('data/train.tsv', engine='python')
    test_set = pd.read_table('data/test.tsv', engine='python')
    return train_set, test_set


# CLEANING AND PREPPING THE DATA
# Handle missing values
def handle_missing(datasets):
    for ds in datasets:
        ds.category_name.fillna(value="nvt", inplace=True)
        ds.brand_name.fillna(value="nvt", inplace=True)
        ds.item_description.fillna(value="nvt", inplace=True)


# Process categorical values
def process_categorical(train_set, test_set):
    # Features that are categorical will be transformed to a numerical value.
    # This is done by training a label encoder on the words in the train and
    # test sets so the labels are the same for both sets. This is applied on these columns:
    # - category_name
    # - brand_name
    le = LabelEncoder()
    le.fit(np.hstack([train_set.category_name, test_set.category_name]))
    train_set.category_name = le.transform(train_set.category_name)
    test_set.category_name = le.transform(test_set.category_name)

    le.fit(np.hstack([train_set.brand_name, test_set.brand_name]))
    train_set.brand_name = le.transform(train_set.brand_name)
    test_set.brand_name = le.transform(test_set.brand_name)

    del le


# Process text fields
def process_text(train_set, test_set):
    # A tokenizer is trained on the combined fields of <i>name</i> and <i>item_description</i>
    def create_tokenizer():
        print("\nTraining tokenizer from scratch")
        # Combine text fields
        text = np.hstack([train_set.name.str.lower(), train_set.item_description.str.lower()])

        # Train the tokenizer
        t = Tokenizer()
        t.fit_on_texts(text)

        with open('Tokenizer/tokenizer.pickle', 'wb') as handle:
            pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return t

    if os.path.isfile('Tokenizer/tokenizer.pickle'):
        with open('Tokenizer/tokenizer.pickle', 'rb') as handle:
            tok = pickle.load(handle)
        print("\nLoaded pre-trained tokenizer")
    else:
        tok = create_tokenizer()

    # Convert text to sequences
    train_set["seq_name"] = tok.texts_to_sequences(train_set.name.str.lower())
    test_set["seq_name"] = tok.texts_to_sequences(test_set.name.str.lower())
    train_set["seq_item_description"] = tok.texts_to_sequences(train_set.item_description.str.lower())
    test_set["seq_item_description"] = tok.texts_to_sequences(test_set.item_description.str.lower())

    del tok


# Scaling the target variable (price)
def scale_target(train_set):
    train_set["target"] = np.log(train_set.price + 1)

    # This target scaler is translating prices to target values and vice-versa.
    # We need to keep it to be able to convert our predictions to prices later on!
    ts = MinMaxScaler(feature_range=(-1, 1))

    train_set["target"] = ts.fit_transform(train_set.target.values.reshape(-1, 1))

    return ts


# THE MODEL FUNCTION
def get_model(dataset, learning_rate=0.002, beta_1=0.5, metrics=None):
    # learning_rate, alpha=0.2, beta1=0.5
    dr_r = 0.5  # Dropout rate

    # Inputs
    name = Input(shape=[dataset.train_x['name'].shape[1]], name='name')
    item_desc = Input(shape=[dataset.train_x['item_desc'].shape[1]], name='item_desc')
    brand_name = Input(shape=[1], name='brand_name')
    category_name = Input(shape=[1], name='category_name')
    item_condition = Input(shape=[1], name='item_condition')
    shipping = Input(shape=[1], name='shipping')

    # Embedding
    emb_name = Embedding(dataset.text_voc_size, 50)(name)
    emb_item_desc = Embedding(dataset.text_voc_size, 50)(item_desc)
    emb_brand_name = Embedding(dataset.brand_voc_size, 10)(
        brand_name)  # Learn to place brand 'reputations' in the same cluster
    emb_category_name = Embedding(dataset.category_voc_size, 10)(category_name)
    # emb_item_condition

    # RNN layers
    RNN_1 = GRU(8)(emb_name)
    RNN_2 = GRU(16)(emb_item_desc)

    # Main layers
    main_layer = concatenate([
        Flatten()(emb_brand_name),
        Flatten()(emb_category_name),
        item_condition,
        RNN_1,
        RNN_2,
        shipping  # this will only have + x$ impact on a price, add this layer more towards the output?
    ])
    main_layer = Dropout(dr_r)(Dense(128)(main_layer))
    main_layer = Dropout(dr_r)(Dense(64)(main_layer))

    # Output
    output = Dense(1, activation='linear')(main_layer)

    # Model
    model = Model([name,
                   item_desc,
                   brand_name,
                   category_name,
                   item_condition,
                   shipping],
                  output)
    opt = Adam(lr=learning_rate, beta_1=beta_1)
    model.compile(loss='mse', optimizer=opt, metrics=metrics)

    return model


# THE LOSS FUNCTION
def rmsle_cust(y_true, y_pred):
    first_log = k.log(k.clip(y_pred, k.epsilon(), None) + 1.)
    second_log = k.log(k.clip(y_true, k.epsilon(), None) + 1.)
    return k.sqrt(k.mean(k.square(first_log - second_log), axis=-1))


# THE TRAINING FUNCTION
def train_model(model, dataset, epochs=2, batch_size=512, verbose=1, filepath='Checkpoints/'):
    callbacks = [
        EarlyStopping('val_loss', patience=2, verbose=verbose),
        ModelCheckpoint(filepath, monitor='val_loss', verbose=verbose, save_best_only=True)
    ]
    # To pass data into multiple input layers for training, train_x hais a dict where the names comply with
    # the input layers
    history = model.fit(dataset.train_x, dataset.train_y, batch_size=batch_size,
                        epochs=epochs, verbose=verbose, callbacks=callbacks,
                        validation_data=(dataset.valid_x, dataset.valid_y),
                        shuffle=dataset.shuffle)
    return history


# THE EVALUATION FUNCTION
def evaluate(model, dataset):
    # Evaluate the model on the dev test (Kaggle)
    val_preds = model.predict(dataset.valid_x)
    val_preds = dataset.scaler.inverse_transform(val_preds)
    val_preds = np.exp(val_preds) - 1

    # Mean absolute error, mean squared error
    labels = dataset.valid_prices
    predictions = val_preds[:, 0]
    rmsle_value = rmsle(labels, predictions)
    print("RMSLE error on the dev test:", str(rmsle_value))


# THE SUBMISSION FUNCTION
def create_submission(model, dataset, batch_size):
    # Make predictions
    preds = model.predict(dataset.test_x, batch_size=batch_size)
    preds = dataset.scaler.inverse_transform(preds)
    preds = np.exp(preds) - 1

    # Create submission file
    submission = dataset.test_ids
    submission['price'] = preds
    submission.to_csv('Submission/submission.csv', index=False)
    submission.price.hist()

if __name__ == '__main__':
    # RUNNING THE CODE
    # Setting up the dataset
    REUSE_DATASET = False

    print('Creating dataset...', end='')
    if REUSE_DATASET and os.path.isfile('Dataset/dataset.pickle'):
        with open('Dataset/dataset.pickle', 'rb') as handle:
            mercari_dataset = pickle.load(handle)
        print('Loaded pre-defined dataset')
    else:
        print('\n\tCreating dataset from scratch...', end='')
        train, test = load_data()
        handle_missing([train, test])
        process_categorical(train, test)
        process_text(train, test)
        target_scaler = scale_target(train)

        # Dataset parameters
        VAL_FRAC = 0.05
        MAX_NAME_LEN = 10  # From studying a histogram of the data
        MAX_DESC_LEN = 75  # From studying a histogram of the data
        TEXT_VOC_SIZE = np.max([np.max(train.seq_name.max()),
                                np.max(test.seq_name.max()),
                                np.max(train.seq_item_description.max()),
                                np.max(test.seq_item_description.max())
                                ])  # + 2  # Why +2?
        CATEGORY_VOC_SIZE = np.max([train.category_name.max(), test.category_name.max()])  # + 1  # Why +1?
        BRAND_VOC_SIZE = np.max([train.brand_name.max(), test.brand_name.max()])  # + 1  # Why +1?
        # max_condition = np.max([train.item_condition_id.max(), test.item_condition_id.max()])# + 1  # Why +1?

        mercari_dataset = Dataset(train, test, scaler=target_scaler, val_frac=VAL_FRAC, max_name_len=MAX_NAME_LEN,
                                  max_desc_len=MAX_DESC_LEN, text_voc_size=TEXT_VOC_SIZE, brand_voc_size=BRAND_VOC_SIZE,
                                  category_voc_size=CATEGORY_VOC_SIZE)

        with open('Dataset/dataset.pickle', 'wb') as h:
            pickle.dump(mercari_dataset, h, protocol=pickle.HIGHEST_PROTOCOL)
        print('Stored dataset for later use')

    # Setting hyper parameters
    REUSE_MODEL = False
    EPOCHS = 5
    BATCH_SIZE = 512
    VERBOSE = 2
    LEARNING_RATE = 0.002
    BETA_1 = 0.5
    METRICS = ['mae', rmsle_cust]
    FILEPATH = 'Checkpoints/base_model.h5'  # 'Checkpoints/base_model.epoch{epoch:02d}.h5'

    # Data analysis
    # Investigation of the dataset to determine some of the hyper parameters

    # # Max sequence length in each field
    # max_name_len = np.max([np.max(train.seq_name.apply(lambda x:len(x))), \
    #                       np.max(test.seq_name.apply(lambda x:len(x)))])
    # max_desc_len = np.max([np.max(train.seq_item_description.apply(lambda x:len(x))), \
    #                       np.max(test.seq_item_description.apply(lambda x:len(x)))])

    # print("Length of longest sequence in the name field:", str(max_name_len))
    # print("Length of longest sequence in the description field:", str(max_desc_len))

    # Setting up the model

    # Creating the model
    print('Creating model...', end='')
    if REUSE_MODEL and os.path.isfile(FILEPATH):
        my_model = load_model(FILEPATH)
        print('Loaded pre-trained model')
    else:
        print('\n\tCreating model from scratch...', end='')
        my_model = get_model(mercari_dataset, learning_rate=LEARNING_RATE, beta_1=BETA_1, metrics=METRICS)
        print('Done')

    # Training the model
    print('Training model...', end='')
    my_history = train_model(my_model, mercari_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE,
                             filepath=FILEPATH)
    print('Done')

    # Evaluating the model
    print('Evaluating model...', end='')
    evaluate(my_model, mercari_dataset)

    # Creating submission file
    # print('Creating submission file...', end='')
    # create_submission(my_model, mercari_dataset, BATCH_SIZE)
    # print('Done')
