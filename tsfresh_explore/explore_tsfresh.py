import pandas as pd
import tsfresh
from tsfresh.utilities.dataframe_functions import make_forecasting_frame

# df = pd.read_csv('../data/input/international-airline-passengers.csv', index_col=0)
# df.index = range(len(df))

df = pd.DataFrame().from_dict({'y': [1, 2, 3, 4]})
print(df)

x, y = make_forecasting_frame(
    x=df['y'],
    kind='ts',
    max_timeshift=1,
    rolling_direction=1
)

print(x)

print(y)