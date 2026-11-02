import numpy as np
import pandas as pd

# Split boundaries refer to evaluated return-ending dates. 
# The input includes 31 preceding price rows for a 30-return observation and a decision before the first evaluated return.
def create_split(prices, start, end, window):
    mask = (prices.index >= start) & (prices.index < end)

    #all the days need to be evaluated
    eval_pos = np.flatnonzero(mask)

    if len(eval_pos) == 0:
        raise ValueError("No trading dates in this period")

    first_pos = eval_pos[0]
    last_pos = eval_pos[-1]

    #we need 30 days of historical return info to make decision
    context_start = first_pos - (window + 1)
    if(context_start < 0):
        raise ValueError("The historical context is not enough")

    return prices.iloc[context_start:last_pos + 1]