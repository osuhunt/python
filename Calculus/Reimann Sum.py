# Using Reimann sums to approximate the area under the standard normal PDF

import random
import numpy as np
import math
import pandas as pd
import scipy.stats as stats

def riemann_sum(lower, upper, num_recs):
    area = 0
    width = (upper-lower)/num_recs
    for i in range(1, num_recs+2, 2):
        midpoint = lower+((i*width)/2)
        height = stats.norm.pdf(midpoint)
        area += width*height
    return area

Area = riemann_sum(1.96, 10, 1000)
print('Area under Normal Curve Above 1.96: ~{}'.format(round(Area,3)))
# Area under Normal Curve Above 1.96: ~0.025
