import skimage
import pickle
import numpy as np
from L_layer_NN_model import predict

#
# LOAD LEARNED PARAMETERS
#
with open("learned_parameters.pickle", "rb") as handle:
    model_parameters = pickle.load(handle)

#
# PREDICT
#
image = skimage.data.imread("image.png", as_gray=True)
image = skimage.transform.rescale(image, 1 / 4, anti_aliasing=True, multichannel=False)
image = np.reshape(image, (image.shape[0] * image.shape[1], 1))
image = image / np.amax(image)
Y_pedicted = predict(image, model_parameters,as_binary=False)
print(Y_pedicted)