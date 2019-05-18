from __future__ import division
from scipy.misc import derivative
from scipy.optimize import fminbound as fmin
from prettytable import PrettyTable as PT
import matplotlib.pyplot as plt 
import warnings
import numpy as np
import decimal as dm


warnings.filterwarnings("ignore")

f = lambda x : (x+np.log( np.sin((x**2)+np.arctan(np.sqrt(1+(x**3))))+1))*x
f_1 = lambda x: derivative(f,x,n=1)
f_4 = lambda x: derivative(f,x,n=4,order=5)

error = lambda x: 10**(-x)

def M(lower,upper,simpson=True):
    if simpson:
        return abs(f(fmin(lambda x: -f_4(x),lower,upper)))
    else:
        return abs(f(fmin(lambda x: -f_1(x),lower,upper)))


def simpson(a,b,errors):
    table = PT()
    table.title = "Numerical Approximation for Integration using Simpson Rule"
    table.field_names = ["Error","N","Approximation"]
    for e in errors:
        s,n = simpsonRule(a,b,e)
        table.add_row([str(error(e)),str(n),str(dm.Decimal(s))])
    print(table)
    print("\n"*3)
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

def determineNSimpson(a,b,e):
    m = M(a,b)
    v = (np.power((b-a),5)*m)/(e*180)
    return int(np.power(v,(1./4.)))


def rectangle(a,b,k,d=True):
    m = M(a,b,simpson=False)
    table = PT()
    table.title = "Numerical Approximation for Integration using Rectangles"
    table.field_names = ["N","Approximation","Error"]
    for i in range(1,k+1):
        s,e = rectangleRule(a,b,i,m,d);
        er = '%.2E' %dm.Decimal(str(e))
        table.add_row([str(i),str(dm.Decimal(s)),er])
    print(table)
    print("\n"*3)
    
def rectangleRule(a,b,n,m,d=True):
    area = lambda c,d: (d-c)*(f(c)+f(d))/2
    v = np.linspace(a,b,n)
    s = 0.0
    for i in range(1,len(v)):
        s += area(v[i-1],v[i])
    e = (np.power((b-a),2)/(n*2))*m
    if d:
        draw(a,b,n);
    return s,e

def draw(a,b,n,rec=True):
    v = np.linspace(a,b,n)
    plt.plot(v,f(v),color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Numerical approximation: '+ ('Rectangular' if rec else 'Simpson'))
    if rec:   
        for i in range(1,len(v)):
            c = v[i-1]
            d = v[i]
            plt.plot([c,c],[0,f((c+d)/2)],color='blue')
            plt.plot([d,d],[0,f(d)],color='blue')
            plt.plot([c,d],[f((c+d)/2),f((c+d)/2)],color='blue')
        plt.show()


simpson(0,1,[7,12])
rectangle(0,1,20,d=False)
