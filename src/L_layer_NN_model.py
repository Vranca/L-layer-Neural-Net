import numpy as np
import sys
import pickle

def sigmoid(z, derivative = False):
    """Computes elemntwise sigmoid function
    
    Arguments:
        z {ndarray} -- Input array
    
    Keyword Arguments:
        derivative {bool} -- Will the output be a derivative (default: {False})
    
    Returns:
        ndarray -- values of sigmoid function or its derivative elementwise
    """
    z = np.clip(z, -709, 709)
    s =  np.exp(z) / (1 + np.exp(z))
    if derivative:
        s = s * ( 1 - s )
    return s

def ReLU(z, derivative = False):
    """Return value max(0,z)
    
    Arguments:
        z {ndarray} -- Input array or number
    
    Keyword Arguments:
        derivative {bool} -- Should the function return the derivative (default: {False})
    
    Returns:
        {ndarray} -- Outputs max(0,z) or the derivative
    """

    if derivative:
        return np.where(z > 0, 1, 0)
    else:
        return np.maximum(0,z)

def leaky_ReLU(z,leak_value=0.1, derivative=False):
    if derivative:
        return np.where(z > 0, 1, leak_value)
    else:
        return np.maximum(z * leak_value,z)

def initialize_parameters(layer_dims):
    """Initializes weights and biases according to input dimensions
    
    Arguments:
        layer_dims {array} -- input_dims, hidden_dims ..., output_dims
    
    Returns:
        Dictionary -- Values Wl and bl, 1 <= l <= L - 1, weights and biases respectively  
    """
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        params["W" + str(l)] = np.random.uniform(low=-1, high=1, size=(layer_dims[l], layer_dims[l-1]))
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params

def forward_activation(A_prev, W, b, activation):

    Z = np.dot(W,A_prev) + b

    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu" or activation == "ReLU":
        A = ReLU(Z)
    elif activation == "leaky ReLU" or activation == "leaky relu":
        A = leaky_ReLU(Z)

    cache = ( A_prev, W, b, Z)

    return A, cache

def forward_propagation(X, params):
    """Executes forward propagation algorithm
    
    Arguments:
        X {ndarray} -- Inputs
        params {dictionary} -- Weights and biases Wl and bl
    
    Returns:
        ndarray, list -- Returns output layer values and a list of caches containing A-prev, W, b, Z
    """

    caches = []
    A = X
    L = len(params) // 2

    for l in range(1,L):
        A_prev = A
        A,cache = forward_activation(A_prev, params["W" + str(l)], params["b" + str(l)], "sigmoid")
        caches.append(cache)

    AL,cache = forward_activation(A, params["W" + str(L)], params["b" + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y, params, lambda_param):

    m = Y.shape[1]
    L = len(params) // 2
    regularization_param = 0

    for l in range(1, L + 1):
        regularization_param += np.sum(params["W" + str(l)] ** 2,keepdims=True)
    regularization_param = regularization_param * (lambda_param/(2*m))

    J = -( 1 / m ) * ( np.sum(Y * np.clip(np.log(np.clip(AL, sys.float_info.min, None)), -sys.float_info.max, sys.float_info.max) + (1 - Y) * np.clip(np.log(np.clip((1 - AL), sys.float_info.min, None)), -sys.float_info.max, sys.float_info.max), keepdims=True)) + regularization_param
    J = np.squeeze(J)

    return J

def backward_activation(dA, cache, activation):

    A_prev, W, b, Z = cache
    m = A_prev.shape[1]

    if activation == "sigmoid":
        dZ = dA * sigmoid(Z, derivative=True)
    elif activation == "relu" or activation == "ReLU":
        dZ = dA * ReLU(Z, derivative=True)
    elif activation == "leaky ReLU" or activation == "leaky relu":
        dZ = dA * leaky_ReLU(Z, derivative=True)

    dW = np.dot(dZ,A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, db

def backpropagation(AL, Y, caches):

    grads = {}
    L = len(caches)
    #m = AL.shape[1]

    dAL = -( (Y / np.clip(AL, sys.float_info.min, None)) - (1 - Y) / np.clip((1 - AL), sys.float_info.min, None))
    #dAL = AL - Y
    current_cache = caches[L-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_activation(dAL, current_cache, "sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = backward_activation(grads["dA" + str(l + 1)], current_cache, "sigmoid")

    return grads

def update_parameters(params, grads, learning_rate, lambda_param):

    L = len(params) // 2

    for l in range(1, L + 1):
        params["W" + str(l)] = (1 - learning_rate * lambda_param / 2) * params["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return params

def train_L_layer_model(X, Y, layer_dims, learning_rate = 0.0075, lambda_param = 0, max_iterations = 3000, print_cost_every_iter = 0, load_parameters=False):

    costs = []

    """
    For randomizing input and output

    perm = np.random.permutation(X.shape[0])
    X = X.T[perm].T
    Y = Y.T[perm].T
    
    """

    if load_parameters:
        with open("learned_parameters.pickle", "rb") as handle:
            params = pickle.load(handle)
    else:
        params = initialize_parameters(layer_dims)

    for i in range(max_iterations):
        
        AL, caches = forward_propagation(X, params)     
        cost = compute_cost(AL, Y, params, lambda_param)
        grads = backpropagation(AL, Y, caches)
        params = update_parameters(params, grads, learning_rate, lambda_param)

        if print_cost_every_iter and i % print_cost_every_iter == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost_every_iter and i % print_cost_every_iter == 0:
            costs.append(cost)

    return params, costs

def predict(X, params, as_binary=False):
    Y_predicted, caches = forward_propagation(X, params)
    if as_binary:
        Y_predicted[Y_predicted > 0.5] = 1
        Y_predicted[Y_predicted <= 0.5] = 0
    return Y_predicted

