#from pkg_resources import resource_stream

import pandas as pd
try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files
    
def load_rain():
    data_file = files(__name__) / 'packageData' / 'rain.dat'
    with data_file.open('r') as stream:
        df = pd.read_csv(stream, index_col=0)
    return df