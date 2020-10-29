import numpy as np
import support_functions as sf


inputsize = 8
hiddenlayersize = 3
outputsize = inputsize

testinput = np.array(([0,0,0,0,1,0,0,0],
                     [0, 0, 1, 0, 0, 0, 0, 0])).reshape(inputsize, 2)

weights1 = np.random.rand(inputsize,hiddenlayersize)
bias1 = np.random.rand(hiddenlayersize,1)

z1 = sf.z_calculation(weights1, testinput, bias1)
a1 = sf.activation_function(z1)

weights2 = np.random.rand(hiddenlayersize, outputsize)
bias2 = np.random.rand(outputsize,1)

z2 = sf.z_calculation(weights2, a1, bias2)
a2 = sf.activation_function(z2)

print(testinput)
print(a2)
