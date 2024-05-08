from pkg_resources import resource_stream
import pandas as pd

def load_rain():
    stream = resource_stream(__name__, 'packageData/rain.dat')
    df = pd.read_csv(stream, index_col=0)
    return(df)