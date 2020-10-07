import glob 
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from L_layer_NN_model import train_L_layer_model
import skimage

m = 0
for i in range(10):
    for filename in glob.glob("My dataset/" + str(i) + "/*.png"):
        m += 1

X = np.empty((1024, m))
Y = np.zeros((10, m))

os.system("cls")
print("Loading data...")
j = 0
for i in range(10):
    for filename in glob.glob("My dataset/" + str(i) + "/*.png"):
        image = skimage.data.imread(filename, as_gray=True)
        image = skimage.transform.rescale(image, 1 / 4, anti_aliasing=True, multichannel=False)
        image = np.reshape(image, (image.shape[0] * image.shape[1], 1))
        image = image / np.amax(image)
        X[:,j] = image[:,0]
        Y[i,j] = 1
        j += 1

        if j % (round(m / 10)) == 0:
            os.system("cls")
            print("Loading data...")
            print(str(100*j//m) + "% of data loaded")
        

os.system("cls")
print("Data loaded.")
print("----------------")

layer_dims = [X.shape[0], 50, 10]

print("Training model...")
model_parameters,costs = train_L_layer_model(X, Y, layer_dims, learning_rate=0.001, lambda_param=0.1, max_iterations=10000, print_cost_every_iter=100, load_parameters=False)
print("Training complete.")

print("Saving data...")
with open("learned_parameters.pickle", "wb+") as handle:
    pickle.dump(model_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Data saved.\n----------------")

plt.plot(costs)
plt.show()