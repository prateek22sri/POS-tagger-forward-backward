import numpy as np
from pprint import pprint

text = "a myth is female moth"
initp = np.array([.45, .35, .15, .05])
tp = np.array([[.03, .42, .5, .05], [.01, .25, .65, .09], [.07, .03, .15, .75], [.3, .25, .15, .3]])
ep = np.array([[.84, .05, .03, .05], [.01, .1, .45, .1], [.02, .02, .02, .6], [.01, .7, .25, .05], [.12, .13, .25, .2]])

# forward algo
forward = np.zeros((5, 4))
for t in range(0, len(text.split())):
    if t == 0:
        tmp = np.dot(initp, tp)
        forward[t] = tmp * ep[t]
    else:
        tmp = np.dot(forward[t - 1], tp)
        forward[t] = tmp * ep[t]
# pprint(forward)

# backward algo
backward = np.zeros((5, 4))
inp = np.array([1, 1, 1, 1])
l = len(text.split()) - 1
for t in range(l, -1, -1):
    if t == l:
        tmp = np.dot(inp, tp)
        a = tmp * ep[t]
        backward[t] = a / np.sum(a)
    else:
        tmp = np.dot(backward[t + 1], tp)
        a = tmp * ep[t]
        backward[t] = a / np.sum(a)
# pprint(backward)

# gamma matrix after smoothing
gamma = np.zeros((5, 4))
gamma = forward * backward
for t in range(0, len(gamma)):
    gamma[t] = gamma[t] / sum(gamma[t])
# pprint(gamma)

print(forward[3][2]) # alpha4(NN)
print(forward[2][3]) # alpha3(VB)
print(forward[0][0]) # alpha4(NN)
print(backward[3][2]) # beta4(NN)
print(backward[1][2]) # beta2(NN)