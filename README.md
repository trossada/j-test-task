# Feature preprocessing (test-task)

There are two files in the resource/ directory: test.tsv, train.tsv. 
Each line of the file is features for one item (job).
The file has two columns:
- id_job - (integer) job id
- features - (string) concatenation of the job features (separated via ','); the first element of the list is the feature code (for example: 2), other elements are the numerical characteristics of this feature.

Module generated a file test_proc.tsv which contains the following set of columns for each job from test.tsv:
- id_job - (integer) vacancy identifier;
- feature_2_stand_{i} - (double) result of standardization (z-score normalization) of the input feature_2_{i};
- max_feature_2_index - (integer) index i of the maximum attribute value feature_2_ {i} for a given vacancy;
- max_feature_2_abs_mean_diff - (double) absolute deviation of the attribute with the index max_feature_2_index from its mean value (feature_2_ {max_feature_2_index}) 

Note:
- parameter `chunk_size` can be used to process large data
- `feature_id` is parameter

To run:

	>>> python3.7 main.py <test-data-source> <train-data-source> <result-source> <feature-id> <chunk_size>

For example:

	>>> python3.7 main.py test.tsv train.tsv test_proc.tsv 2 100

To run tests:

	>>> python -m unittest tests/test_utils.py 
