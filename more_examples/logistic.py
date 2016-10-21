import theano
import theano.tensor as T

x=T.dmatrix('x')
s=1/(1+T.exp(-x))
logistic=theano.function([x],s)

print(logistic([[0,1],[-1,-2]]))
# ==>[[ 0.5         0.73105858]
#	 [ 0.26894142  0.11920292]]


s2=(1+T.tanh(x/22))/2
logistic2=theano.function([x],s2)

print(logistic2([[0,1],[-1,-2]]))