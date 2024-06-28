import sys
from pathlib import Path
import pytest
import pandas as pd

# Ensure the project path is in the sys.path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Import the necessary modules and functions
from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bike_sharing_pipe
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.validataion import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bike_sharing_pipe = load_pipeline(file_name=pipeline_file_name)

from bikeshare_model.predict import make_prediction  # Assuming make_prediction is in predict.py

@pytest.fixture
def sample_input_data():
    return {
        'dteday': ['2012-11-05'],
        'season': ['winter'],
        'hr': ['6am'],
        'holiday': ['No'],
        'weekday': ['Mon'],
        'workingday': ['Yes'],
        'weathersit': ['Mist'],
        'temp': [6.1],
        'atemp': [3.0014],
        'hum': [49.0],
        'windspeed': [19.0012],
        'casual': [4],
        'registered': [135],
        'cnt': [139]
    }

def test_make_prediction(sample_input_data):
    # Convert the sample input data to a DataFrame
    input_df = pd.DataFrame(sample_input_data)
    
    # Make a prediction
    result = make_prediction(input_data=input_df)
    
    # Check that the result is a dictionary
    assert isinstance(result, dict)
    
    # Check that the result contains 'predictions', 'version', and 'errors'
    assert 'predictions' in result
    assert 'version' in result
    assert 'errors' in result
    
    # Check that the 'version' is correct
    assert result['version'] == _version
    
    # Check that there are no errors
    assert result['errors'] is None or len(result['errors']) == 0
    
    # Check that the predictions are not None
    assert result['predictions'] is not None
    
    # Check that the predictions have the same length as the input data
    assert len(result['predictions']) == len(input_df)

if __name__ == "__main__":
    pytest.main([__file__])
