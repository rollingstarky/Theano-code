import theano

a=theano.tensor.vector()    #declare variable
b=theano.tensor.vector()	#declare variable
out=(a+b)**2				#build symbolic function					
f=theano.function([a,b],out)	#compile function
print(f([1,2],[4,5]))		# ==> [25. 49.]
