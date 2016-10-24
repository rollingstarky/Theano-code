from theano import shared
import theano.tensor as T
from theano import function

state=shared(0)
inc=T.iscalar('inc')
accumlator=function([inc],state,updates=[(state,state+inc)])

print(state.get_value())
#==> 0
accumlator(1)
print(state.get_value())
#==> 1
accumlator(300)
print(state.get_value())
#==> 301

#reset the state
state.set_value(-1)
accumlator(3)
print(state.get_value())
#==> 2

#define more than one function
decrementor=function([inc],state,updates=[(state,state-inc)])
decrementor(2)
print(state.get_value())
#==> 0
