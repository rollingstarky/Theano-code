#Adding two Scalars

import numpy
import theano.tensor as T
from theano import function

x=T.dscalar('x')
y=T.dscalar('y')

z=x+y
f=function([x,y],z)

print(f(2,3))
# ==> 5.0
print(numpy.allclose(f(16.3,12.1),28.4))
# ==> True


#Adding two Matrices

x1=T.dmatrix('x1')
y1=T.dmatrix('y1')
z1=x1+y1
f1=function([x1,y1],z1)

print(f1([[1,2],[3,4]],[[10,20],[30,40]]))
# ==> 【[11.,22.]
#     [33.,44.]】

