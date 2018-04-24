import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
print("Lets Play with training set:")
index = int(input("Which train set you want?"))
while(index<209):
	plt.imshow(train_x_orig[index])
	print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
	plt.show()
	index = int(input("Which train set you want?"))
print("##################")
print("Lets Play with test set:")
index = int(input("Which test set you want?"))
while(index<55):
	plt.imshow(test_x_orig[index])
	print ("y = " + str(test_y[0,index]) + ". It's a " + classes[test_y[0,index]].decode("utf-8") +  " picture.")
	plt.show()
	index = int(input("Which test set you want?"))
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost

    # START CODE HERE #
    parameters = initialize_parameters_deep(layers_dims)
    # END CODE HERE #

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        # END CODE HERE #

        # Compute cost.
        cost = compute_cost(AL, Y)
        # END CODE HERE #

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
        # END CODE HERE #

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # END CODE HERE #

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
print_mislabeled_images(classes, test_x, test_y, pred_test)



print("Lets Play with test set:")
index = int(input("Which test set you want?"))
while(index<55):

	my_label_y = test_y[:, index] # the true class of your image (1 -> cat, 0 -> non-cat)
	fname = test_x_orig[index]
	my_image = scipy.misc.imresize(fname, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
	my_predicted_image = predict(my_image, my_label_y, parameters)
	plt.imshow(fname)
	print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
	index = int(input("Which test set you want?"))

## END CODE HERE ##
my_label_y=[0,1,0]
for i in range(2):
	fname = str(i)+".png"
	my_label_y=my_label_y[:,i]
	image = np.array(ndimage.imread(fname, flatten=False))
	my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
	my_predicted_image = predict(my_image, my_label_y, parameters)
	plt.imshow(image)
	plt.show()
	print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
