import neuralnet as nn
import matplotlib.pyplot as plt


learningrate = 5
reglambda = 0.00001
epochs = 5000
weightinitialisation = 0.1

reglambda = 0
costhistory1, accuracyhistory1 = nn.run_nn(weightinitialisation, learningrate, reglambda, epochs)

reglambda = 0.01
costhistory2, accuracyhistory2 = nn.run_nn(weightinitialisation, learningrate, reglambda, epochs)

reglambda = 0.02
costhistory3, accuracyhistory3 = nn.run_nn(weightinitialisation, learningrate, reglambda, epochs)

reglambda = 0.03
costhistory4, accuracyhistory4 = nn.run_nn(weightinitialisation, learningrate, reglambda, epochs)

reglambda = 0.04
costhistory5, accuracyhistory5 = nn.run_nn(weightinitialisation, learningrate, reglambda, epochs)

t = range(0, epochs)
plt.plot(t, costhistory1, 'tab:blue', t, accuracyhistory1, 'tab:blue', label='0')
plt.plot(t, costhistory2, 'tab:orange', t, accuracyhistory2, 'tab:orange', label='0.1')
plt.plot(t, costhistory3, 'tab:green', t, accuracyhistory3, 'tab:green', label='0.2')
plt.plot(t, costhistory4, 'tab:red', t, accuracyhistory4, 'tab:red', label='0.3')
plt.plot(t, costhistory5, 'tab:purple', t, accuracyhistory5, 'tab:purple', label='0.4')
plt.legend(title='lambda')
plt.show()
