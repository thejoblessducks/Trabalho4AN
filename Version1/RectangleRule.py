from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

f = lambda x : (x+np.log( np.sin((x**2)+np.arctan(np.sqrt(1+np.power(x,3))))+1))*x
def rectIntegration(lower,upper,N):
    points = np.linspace(lower,upper,N); #calculate step
    area = 0
    for r in range(len(points)):
        area += f( lower + r*(upper-lower)/N )*((upper-lower)/N)
    return area

print(rectIntegration(0,1,10));