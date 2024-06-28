import pytest
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def sample_dataframe():
    data = {
        'dteday': [datetime(2020, 1, 1), datetime(2020, 1, 2), pd.NaT, datetime(2020, 1, 4)],
        'weekday': [np.nan, 'Wed', np.nan, 'Fri'],
        'weathersit': ['Clear', np.nan, 'Rain', np.nan],
        'temp': [0.3, 0.45, 0.2, 0.6],
        'windspeed': [0.1, 0.2, 0.3, 0.4],
        'workingday': ['Yes', 'No', 'Yes', 'No']
    }
    return pd.DataFrame(data)
