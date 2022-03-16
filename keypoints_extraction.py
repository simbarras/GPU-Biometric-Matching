import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('WebAgg')

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True

def mouse_event(event):
    print('x: {} and y: {}'.format(event.xdata, event.ydata))


fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
x = np.linspace(-10, 10, 100)
y = np.exp(x)

plt.plot(x, y)

plt.show()