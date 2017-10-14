import numpy as np
from pprint import pprint

text = "a myth is a female moth"
initp = np.array([.45, .35, .15, .05])
tp = np.array([[.03, .42, .5, .05], [.01, .25, .65, .09], [.07, .03, .15, .75], [.3, .25, .15, .3]])
ep = np.array([[.84, .05, .03, .05], [.01, .1, .45, .1], [.02, .02, .02, .6], [.84, .05, .03, .05],[.01, .7, .25, .05], [.12, .13, .25, .2]])

# forward algo
forward = np.zeros((6, 4))
for t in range(0, len(text.split())):
    if t == 0:
        tmp = np.dot(initp, tp)
        forward[t] = tmp * ep[t]
    else:
        tmp = np.dot(forward[t - 1], tp)
        forward[t] = tmp * ep[t]
# pprint(forward)

# backward algo
backward = np.zeros((6, 4))
inp = np.array([1, 1, 1, 1])
l = len(text.split()) - 1
for t in range(l, -1, -1):
    if t == l:
        tmp = np.dot(inp, tp)
        backward[t] = tmp * ep[t]
    else:
        tmp = np.dot(backward[t + 1], tp)
        backward[t] = tmp * ep[t]
# pprint(backward)

# gamma matrix after smoothing alpha and beta values
gamma = np.zeros((6, 4))
gamma = forward * backward

print("alpha4(NN) is ", forward[3][2]) # alpha4(NN)
print("alpha3(VB) is ",forward[2][3]) # alpha3(VB)
print("alpha1(DT) is ",forward[0][0]) # alpha1(NN)
print("beta4(NN) is ",backward[3][2]) # beta4(NN)
print("beta2(NN) is ",backward[1][2]) # beta2(NN)