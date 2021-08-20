import numpy as np 

arr = np.array([
    [1], 
    [4],
    [0],
    [5]
])

for x in arr:
    if x > 3:
        print('3')
    else:
        print(x)
