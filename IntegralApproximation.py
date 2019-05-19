from __future__ import division
import warnings

from scipy.misc import derivative
from scipy.optimize import fminbound as fmin

import matplotlib.pyplot as plt 
from prettytable import PrettyTable as PT
from time import localtime, strftime

import numpy as np
import decimal as dm

warnings.filterwarnings("ignore")

'''-----------------------------------------------------------------------------
Functions: f(x),f'(x) and f^4(x) and error function 10^-x
-----------------------------------------------------------------------------'''
f = lambda x : (x+np.log( np.sin((x**2)+np.arctan(np.sqrt(1+(x**3))))+1))*x
f_1 = lambda x: derivative(f,x,n=1)
f_4 = lambda x: derivative(f,x,n=4,order=5)

error = lambda x: 10**(-x)


'''-----------------------------------------------------------------------------
M calculation, max|f'(x)| or max|f^4(x)| for xE]a,b[
-----------------------------------------------------------------------------'''
def M(lower,upper,simpson=True):
    if simpson:
        return abs(f(fmin(lambda x: -f_4(x),lower,upper)))
    else:
        return abs(f(fmin(lambda x: -f_1(x),lower,upper)))


'''-----------------------------------------------------------------------------
Simpson Rule for Integration Approximation
-----------------------------------------------------------------------------'''
#Organize data and display
def simpson(a,b,errors):
    table = PT()
    table.title = "Numerical Approximation for Integration using Simpson Rule"
    table.field_names = ["Error","N","Approximation M="+str(M(a,b))]

    for e in errors:
        s,n = simpsonRule(a,b,e)
        table.add_row([str(error(e)),str(n),str(dm.Decimal(s))])
    print(table)
    print("\n"*3)

#Determine n for error e
def determineNSimpson(a,b,e):
    m = M(a,b)
    v = (np.power((b-a),5)*m)/(e*180)
    v = np.power(v,(1./4.))
    m = int(v)+2 if v%2==0 else int(v)+1
    return m

#Actual method
def simpsonRule(a,b,e):
    n = determineNSimpson(a,b,error(e))
    val = lambda x: 2 if (i%2==0) else 4
    h = (b-a)/n    
    v = np.linspace(a,b,n)
    v = v[1:-1]
    s = 0.0
    for i,x in enumerate(v):
        s += val(i)*f(x)
    return (h/3)*(f(a)+f(b)+s),n


'''-----------------------------------------------------------------------------
Rectangle Rule for Integration Approximation
-----------------------------------------------------------------------------'''
#Organize data and display
def rectangle(a,b,k,d=True):
    #if d is true,it will show the graph of approximation for every k
    m = M(a,b,simpson=False) #m calculation~

    table = PT()
    table.title = "Numerical Approximation for Integration using Rectangles"
    table.field_names = ["N","Approximation M="+str(m),"Error"]

    for i in range(1,k+1):
        s,e = rectangleRule(a,b,2**i,m,d);
        er = '%.2E' %dm.Decimal(str(e))
        table.add_row([str(2**i),str(dm.Decimal(s)),er])
    print(table)
    print("\n"*3)

#Rectangle rule method
def rectangleRule(a,b,n,m,d=True):
    step = (b-a)/n
    s = 0.0
    for i in range(n):
        s += f((a+step/2)+i*step)
    s *= step
    #error calculation
    e = (np.power((b-a),2)/(n*2))*m

    if d and n<=256:#draw option
        draw(a,b,n);
    return s,e
#Draw f and approximation rectangles
def draw(a,b,n):
    v = np.linspace(a,b,n)
    plt.plot(v,f(v),color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Numerical approximation: Rectangular')
    for i in range(1,len(v)):
        c = v[i-1]
        d = v[i]
        plt.plot([c,c],[0,f((c+d)/2)],color='blue')
        plt.plot([d,d],[0,f(d)],color='blue')
        plt.plot([c,d],[f((c+d)/2),f((c+d)/2)],color='blue')
    plt.savefig("./Images/RectangleDraw_N"+str(n)+"_"+strftime("%Y%m%d_%H%M%S", localtime())+".png",dpi=300, bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()



#-------------------------------------------------------------------------------
disp = input("Display Rectangles Graph? (yes/no)")
disp = disp == 'yes'
print
simpson(0,1,[7,12])#apply simpson to error 7 and 12 in [0,1]
rectangle(0,1,20,d=disp) #apply rectangle in [0,1] in n points n=1(1)20
