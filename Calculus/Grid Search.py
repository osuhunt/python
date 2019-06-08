def calculate_SSE(X, Y, slope, intercept):
    error=((slope*X+intercept)-Y)**2
    return sum(error)
    
def Grid_Search(X, Y, intercept_range, slope_range):
    best_slope = slope_range[0]
    best_intercept = intercept_range[0]
    lowest_SSE = calculate_SSE(X, Y, slope = best_slope, intercept = best_intercept)


    for intercept in np.linspace(int(intercept_range[0]), int(intercept_range[1]), 20):
        for slope in np.linspace(int(slope_range[0]), int(slope_range[1]), 20):
            sse = calculate_SSE(X, Y, slope = slope, intercept = intercept)
            if sse < lowest_SSE:
                best_slope = slope
                best_intercept = intercept
                lowest_SSE = sse
      
    return best_slope,best_intercept,lowest_SSE
