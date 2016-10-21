from theano import *
import theano.tensor as T

x,y,w=T.dscalars('x', 'y','w')
z=(x+y)*w
f=function([x,In(y,value=1),In(w,value=2,name='w_by_name')],z)

print(f(33))
# ==>68.0
print(f(33,2))
#==>70.0
print(f(33,0,1))
# ==>33.0
print(f(33,w_by_name=1))
# ==>34.0
print(f(33,w_by_name=1,y=0))
# ==>33.0