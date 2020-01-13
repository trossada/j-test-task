import os
import logging
from sys import argv

import pandas as pd
import numpy as np

from feature_scaler import FeatureScaler
from utils import get_max_feature_index_and_abs_mean_diff

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

FEATURE_TITLE_PATTERN = 'feature_%d_%d'
FEATURE_STAND_PATTERN_TITLE = 'feature_%d_stand_%d'
MAIN_COLUMNS_TITLES = ['id_job', 'feature_id']

tmp_data_source = 'tmp.tsv'

_, train_data_source, test_data_source, result_data_source, feature_id, chunk_size = argv
feature_id, chunk_size = int(feature_id), int(chunk_size)

logger.info(f'Features with id = {feature_id} will be processed')
logger.info(f'Chunk size: {chunk_size}')

feature_scaler = FeatureScaler()

# Process train data (fit scaler)
processed_first_chunk = False
for df in pd.read_csv(train_data_source, sep='\t|,', skiprows=1, header=None, chunksize=chunk_size):
    if not processed_first_chunk:
        n_features = df.shape[1] - 2
        logger.info(f'Feature count: {n_features}')
        feature_titles = [FEATURE_TITLE_PATTERN % (feature_id, i) for i in range(n_features)]
        column_titles = MAIN_COLUMNS_TITLES + feature_titles

    df.columns = column_titles
    X = df[df.feature_id == feature_id][feature_titles].values.astype(float)
    feature_scaler.fit(X)
    processed_first_chunk = True

# Standartize test data
processed_first_chunk = False
for df in pd.read_csv(test_data_source, sep='\t|,', skiprows=1, header=None, chunksize=chunk_size):
    df.columns = column_titles
    X = df[df.feature_id == feature_id][feature_titles].values.astype(float)
    X_feature_stand = feature_scaler.transform(X)

    feature_stand_titles = [FEATURE_STAND_PATTERN_TITLE % (feature_id, i) for i in range(n_features)]
    new_df = pd.DataFrame(data=np.hstack((df.values,  X_feature_stand)), columns=column_titles + feature_stand_titles)
    new_df.to_csv(tmp_data_source,
                  sep='\t',
                  header=not processed_first_chunk,
                  mode='a' if processed_first_chunk else 'w',
                  index=False)
    processed_first_chunk = True

# Calculating max feature index and abs mean diff
processed_first_chunk = False
for df in pd.read_csv(tmp_data_source, sep='\t', chunksize=chunk_size):
    X = df[df.feature_id == feature_id][feature_titles].values.astype(float)
    X_max_feature_index, X_max_feature_abs_mean_diff = get_max_feature_index_and_abs_mean_diff(X, feature_scaler.mean_)
    df.insert(0, 'max_feature_2_index', X_max_feature_index)
    df.insert(0, 'max_feature_2_abs_mean_diff', X_max_feature_abs_mean_diff)
    df = df[MAIN_COLUMNS_TITLES
            + feature_stand_titles
            + ['max_feature_2_index', 'max_feature_2_abs_mean_diff']].astype({_: 'int32' for _ in MAIN_COLUMNS_TITLES})

    df.to_csv(result_data_source,
              sep='\t',
              header=not processed_first_chunk,
              mode='a' if processed_first_chunk else 'w',
              index=False)
    processed_first_chunk = True
os.remove(tmp_data_source)

