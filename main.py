import numpy as np
import support_functions as sf
import matplotlib.pyplot as plt
import tkinter

np.random.seed(1)

inputsize = 8
hiddenlayersize = 3
outputsize = inputsize
weightinitialisation = 0.1
learningrate = 15
reglambda = 0.0001

weights2 = np.random.randn(hiddenlayersize, inputsize) * weightinitialisation
bias2 = np.random.randn(hiddenlayersize, 1) * weightinitialisation

weights3 = np.random.randn(outputsize, hiddenlayersize) * weightinitialisation
bias3 = np.random.randn(outputsize, 1) * weightinitialisation

dweights3 = np.zeros((outputsize, hiddenlayersize))
dbias3 = np.zeros((outputsize, 1))

dweights2 = np.zeros((hiddenlayersize, inputsize))
dbias2 = np.zeros((hiddenlayersize, 1))

testinput = np.identity(inputsize, dtype=int)

# testinput = np.array([[1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0], [0,1,0,0,0,0,0,0]]).T


costhistory = []
weightsumhistory = []
accuracyhistory = []

epochs = 0
a1 = testinput
heatmap = []
h = 0

reglambda = 0.00001
while reglambda < 0.02:
    heatmap.append([])
    print("reglambda: ", reglambda)

    learningrate = 1
    while learningrate < 80:

        converged = False
        epochs = 0
        weights2 = np.random.randn(hiddenlayersize, inputsize) * weightinitialisation
        bias2 = np.random.randn(hiddenlayersize, 1) * weightinitialisation

        weights3 = np.random.randn(outputsize, hiddenlayersize) * weightinitialisation
        bias3 = np.random.randn(outputsize, 1) * weightinitialisation

        dweights3 = np.zeros((outputsize, hiddenlayersize))
        dbias3 = np.zeros((outputsize, 1))

        dweights2 = np.zeros((hiddenlayersize, inputsize))
        dbias2 = np.zeros((hiddenlayersize, 1))
        while not converged:

            epochs += 1
            z2 = sf.z_calculation(weights2, a1, bias2)
            a2 = sf.activation_function(z2)

            z3 = sf.z_calculation(weights3, a2, bias3)
            a3 = sf.activation_function(z3)

            # Forward propagation ends here, now comes backward propagation

            m = a2.shape[1]

            delta3 = a3 * (1 - a3) * (a3 - testinput)
            delta2 = a2 * (1 - a2) * np.dot(weights2, delta3)

            DELTA3 = np.dot(delta3, a2.T)
            DELTA2 = np.dot(delta2, a1.T)

            dweights3 = (DELTA3 + reglambda * weights3) / m
            dbias3 = np.sum(delta3, axis=1, keepdims=True) / m

            dweights2 = (DELTA2 + reglambda * weights2) / m
            dbias2 = np.sum(delta2, axis=1, keepdims=True) / m

            # delta's calculated: now updating weights.

            weights2 = weights2 - learningrate * dweights2
            bias2 = bias2 - learningrate * dbias2

            weights3 = weights3 - learningrate * dweights3
            bias3 = bias3 - learningrate * dbias3

            # cost = sf.calculate_cost(a3, testinput)

            # print("In epoch ", epoch, "the cost is ", cost)
            # costhistory.append(cost)
            # weightsumhistory.append(np.sum(weights2) + np.sum(weights3))

            amountcorrect = 0
            for i in range(0, inputsize):
                if np.argmax(testinput[:, i]) == np.argmax(a3[:, i]):
                    amountcorrect += 1
            accuracyhistory.append(amountcorrect)

            if epochs > 11:
                count = 0
                for last in accuracyhistory[-10:]:

                    if last == 8:
                        count += 1
                if count == 10:
                    converged = True

            if epochs > 99999:
                converged = True
        heatmap[h].append(epochs)
        learningrate = learningrate * 1.5
        print("learning rate: ", learningrate, "epochs: ", epochs)
    h += 1
    reglambda = reglambda * 2

print(heatmap)
heatplot = np.array(heatmap)
reglambdas = [round(0.00001 * (2**rl), 5 )for rl in range(11)]
learningrates = [round(1 * (1.5**lr),5) for lr in range(11)]
plt.xticks(ticks=np.arange(11), labels=learningrates, rotation=90)
plt.yticks(ticks=np.arange(11), labels=reglambdas)
plt.imshow(np.log(heatplot), interpolation="nearest")
plt.show()
