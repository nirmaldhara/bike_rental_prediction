import pytest
from processing.features import WeekdayImputer, WeathersitImputer, OutlierHandler, Mapper, WeekdayOneHotEncoder

def test_weekday_imputer(sample_dataframe):
    imputer = WeekdayImputer(variables='weekday')
    transformed_df = imputer.transform(sample_dataframe)
    assert transformed_df['weekday'].isnull().sum() == 0
    assert 'dteday' not in transformed_df.columns

def test_weathersit_imputer(sample_dataframe):
    imputer = WeathersitImputer(variables='weathersit')
    transformed_df = imputer.transform(sample_dataframe)
    assert transformed_df['weathersit'].isnull().sum() == 0
    assert (transformed_df['weathersit'] == 'Clear').sum() == 2

def test_outlier_handler(sample_dataframe):
    outlier_handler = OutlierHandler(columns=['temp', 'windspeed'])
    outlier_handler.fit(sample_dataframe)
    transformed_df = outlier_handler.transform(sample_dataframe)
    for column in ['temp', 'windspeed']:
        assert (transformed_df[column] >= outlier_handler.lower_bounds_[column]).all()
        assert (transformed_df[column] <= outlier_handler.upper_bounds_[column]).all()

def test_mapper(sample_dataframe):
    mappings = {'Yes': 1, 'No': 0}
    mapper = Mapper(variables='workingday', mappings=mappings)
    transformed_df = mapper.transform(sample_dataframe)
    assert set(transformed_df['workingday'].unique()) == {0, 1}

def test_weekday_one_hot_encoder(sample_dataframe):
    encoder = WeekdayOneHotEncoder(column_name='weekday')
    encoder.fit(sample_dataframe)
    transformed_df = encoder.transform(sample_dataframe)
    assert 'weekday' not in transformed_df.columns
    assert set(encoder.encoded_columns).issubset(transformed_df.columns)
    assert transformed_df[encoder.encoded_columns].sum().sum() == len(sample_dataframe)
