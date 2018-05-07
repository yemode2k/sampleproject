
# coding: utf-8

# The main objective is to predict the customer life time value.

# importing the required libs :

import Algo
import numpy as np
import pandas as pd
import sklearn as sk
import Algo.core.pandasutils as pdu
from Algo.doctor.preprocessing import PCA
from collections import defaultdict, Counter


# And tune pandas display options:

pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

preparation_steps = []
preparation_output_schema = {u'userModified': False, u'columns': [{u'timestampNoTzAsDate': False, u'type': u'string', u'name': u'customer_id', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'age', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'double', u'name': u'price_first_item_purchased', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'string', u'name': u'gender', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'revenue', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'string', u'name': u'join_ip', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'string', u'name': u'join_ip_country', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'double', u'name': u'join_pages_visited', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'boolean', u'name': u'join_campain', u'maxLength': -1}, {u'timestampNoTzAsDate': False, u'type': u'bigint', u'name': u'join_GDP_cap', u'maxLength': -1}]}

ml_dataset_handle = Algo.Dataset('crm_prepared')
ml_dataset_handle.set_preparation_steps(preparation_steps, preparation_output_schema)
get_ipython().magic(u'time ml_dataset = ml_dataset_handle.get_dataframe(limit = 100000)')

print 'Base data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1])
# Five first records",
ml_dataset.head(5)
ml_dataset = ml_dataset[[u'join_GDP_cap', u'join_pages_visited', u'revenue', u'join_campain', u'gender', u'age', u'join_ip', u'join_ip_country', u'price_first_item_purchased']]

# astype('unicode') does not work as expected
def coerce_to_unicode(x):
    if isinstance(x, str):
        return unicode(x,'utf-8')
    else:
        return unicode(x)

categorical_features = [u'join_campain', u'gender', u'join_ip', u'join_ip_country']
numerical_features = [u'join_GDP_cap', u'join_pages_visited', u'age', u'price_first_item_purchased']
text_features = []
from Algo.doctor.utils import datetime_to_epoch
for feature in categorical_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in text_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in numerical_features:
    if ml_dataset[feature].dtype == np.dtype('M8[ns]'):
        ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
    else:
        ml_dataset[feature] = ml_dataset[feature].astype('double')
ml_dataset['__target__'] = ml_dataset['revenue']
del ml_dataset['revenue']
# Remove rows for which the target is unknown.
ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

# The dataset needs to be split into 2 new sets, one that will be used for training the model (train set)
# and another that will be used to test its generalization capability (test set)

train, test = pdu.split_train_valid(ml_dataset, prop=0.9)
print 'Train data has %i rows and %i columns' % (train.shape[0], train.shape[1])
print 'Test data has %i rows and %i columns' % (test.shape[0], test.shape[1])


# Preprocessing

drop_rows_when_missing = []
impute_when_missing = [{'impute_with': u'MEAN', 'feature': u'join_GDP_cap'}, {'impute_with': u'MEAN', 'feature': u'join_pages_visited'}, {'impute_with': u'MEAN', 'feature': u'age'}, {'impute_with': u'MEAN', 'feature': u'price_first_item_purchased'}]

# Features for which we drop rows with missing values"
for feature in drop_rows_when_missing:
    train = train[train[feature].notnull()]
    test = test[test[feature].notnull()]
    print 'Dropped missing records in %s' % feature

# Features for which we impute missing values"
for feature in impute_when_missing:
    if feature['impute_with'] == 'MEAN':
        v = train[feature['feature']].mean()
    elif feature['impute_with'] == 'MEDIAN':
        v = train[feature['feature']].median()
    elif feature['impute_with'] == 'CREATE_CATEGORY':
        v = 'NULL_CATEGORY'
    elif feature['impute_with'] == 'MODE':
        v = train[feature['feature']].value_counts().index[0]
    elif feature['impute_with'] == 'CONSTANT':
        v = feature['value']
    train[feature['feature']] = train[feature['feature']].fillna(v)
    test[feature['feature']] = test[feature['feature']].fillna(v)
    print 'Imputed missing values in feature %s with value %s' % (feature['feature'], unicode(str(v), 'utf8'))


# We can now handle the categorical features (still using the settings defined in Models):

# Let's dummy-encode the following features.
# A binary column is created for each of the 100 most frequent values.

LIMIT_DUMMIES = 100

categorical_to_dummy_encode = [u'join_campain', u'gender', u'join_ip', u'join_ip_country']

# Only keep the top 100 values
def select_dummy_values(train, features):
    dummy_values = {}
    for feature in categorical_to_dummy_encode:
        values = [
            value
            for (value, _) in Counter(train[feature]).most_common(LIMIT_DUMMIES)
        ]
        dummy_values[feature] = values
    return dummy_values

DUMMY_VALUES = select_dummy_values(train, categorical_to_dummy_encode)

def dummy_encode_dataframe(df):
    for (feature, dummy_values) in DUMMY_VALUES.items():
        for dummy_value in dummy_values:
            dummy_name = u'%s_value_%s' % (feature, unicode(dummy_value))
            df[dummy_name] = (df[feature] == dummy_value).astype(float)
        del df[feature]
        print 'Dummy-encoded feature %s' % feature

dummy_encode_dataframe(train)

dummy_encode_dataframe(test)
rescale_features = {u'age': u'AVGSTD', u'join_GDP_cap': u'AVGSTD', u'join_pages_visited': u'AVGSTD', u'price_first_item_purchased': u'AVGSTD'}
for (feature_name, rescale_method) in rescale_features.items():
    if rescale_method == 'MINMAX':
        _min = train[feature_name].min()
        _max = train[feature_name].max()
        scale = _max - _min
        shift = _min
    else:
        shift = train[feature_name].mean()
        scale = train[feature_name].std()
    if scale == 0.:
        del train[feature_name]
        del test[feature_name]
        print 'Feature %s was dropped because it has no variance' % feature_name
    else:
        print 'Rescaled %s' % feature_name
        train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
        test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale


# Modeling
# Before actually creating our model, we need to split the datasets into their features and labels parts:

train_X = train.drop('__target__', axis=1)
test_X = test.drop('__target__', axis=1)

train_Y = np.array(train['__target__'])
test_Y = np.array(test['__target__'])


# Now we can finally create our model !

from sklearn.linear_model import RidgeCV
clf = RidgeCV(fit_intercept=True, normalize=True)

get_ipython().magic(u'time clf.fit(train_X, train_Y)')


# Build up our result dataset

# In[ ]:


get_ipython().magic(u'time _predictions = clf.predict(test_X)')
predictions = pd.Series(data=_predictions, index=test_X.index, name='predicted_value')

# Build scored dataset
results_test = test_X.join(predictions, how='left')
results_test = results_test.join(test['__target__'], how='left')
results_test = results_test.rename(columns= {'__target__': 'revenue'})

# model's accuracy:

c =  results_test[['predicted_value', 'revenue']].corr()
print 'Pearson correlation: %s' % c['predicted_value'][1]

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100,
    random_state=1337,
    max_depth=13,
    min_samples_leaf=10,
    verbose=2)

get_ipython().magic(u'time clf.fit(train_X, train_Y)')


get_ipython().magic(u'time _predictions = clf.predict(test_X)')
predictions = pd.Series(data=_predictions, index=test_X.index, name='predicted_value')

# Build scored dataset
results_test = test_X.join(predictions, how='left')
results_test = results_test.join(test['__target__'], how='left')
results_test = results_test.rename(columns= {'__target__': 'revenue'})


#  feature importances

feature_importances_data = []
features = train_X.columns
for feature_name, feature_importance in zip(features, clf.feature_importances_):
    feature_importances_data.append({
        'feature': feature_name,
        'importance': feature_importance
    })

# Plot the results
pd.DataFrame(feature_importances_data)    .set_index('feature')    .sort_values(by='importance')[-10::]    .plot(title='Top 10 most important variables',
          kind='barh',
          figsize=(10, 6),
          color='#348ABD',
          alpha=0.6,
          lw='1',
          edgecolor='#348ABD',
          grid=False,)


# YModel's accuracy:

c =  results_test[['predicted_value', 'revenue']].corr()
print 'Pearson correlation: %s' % c['predicted_value'][1]


