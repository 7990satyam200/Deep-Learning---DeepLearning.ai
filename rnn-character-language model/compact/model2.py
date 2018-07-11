import numpy as np
import random

data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

with open("dinos.txt") as f:
    examples = f.readlines()
examples = [x.lower().strip() for x in examples]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def model(data, ix_to_char, char_to_ix, learning_rate=0.01, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):

    n_x, n_y = vocab_size, vocab_size

    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}

    loss1 = -np.log(1.0/vocab_size)*dino_names
    np.random.seed(0)
    np.random.shuffle(examples)
    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))
    # Optimization loop
    for j in range(num_iterations):

        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]
        x, a, y_hat = {}, {}, {}
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
        a[-1] = np.copy(a_prev)
        # initialize your loss to 0
        loss = 0
        for t in range(len(X)):
            x[t] = np.zeros((vocab_size,1))
            if (X[t] != None):
                x[t][X[t]] = 1
            a[t]= np.tanh(np.dot(Wax, x[t]) + np.dot(Waa, a[t-1]) + b)
            y_hat[t] =softmax(np.dot(Wya, a[t]) + by)
            # Update the loss by substracting the cross-entropy term of this time-step from it.
            loss -= np.log(y_hat[t][Y[t],0])
        cache = (y_hat, a, x)
        gradients = {}

        # Retrieve from cache and parameters
        (y_hat, a, x) = cache
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
        gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
        gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
        gradients['da_next'] = np.zeros_like(a[0])

        # Backpropagate through time
        for t in reversed(range(len(X))):
            dy = np.copy(y_hat[t])
            dy[Y[t]] -= 1
            #gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
            gradients['dWya'] += np.dot(dy, a[t].T)
            gradients['dby'] += dy
            da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
            daraw = (1 - a[t] * a[t]) * da # backprop through tanh nonlinearity
            gradients['db'] += daraw
            gradients['dWax'] += np.dot(daraw, x[t].T)
            gradients['dWaa'] += np.dot(daraw, a[t-1].T)
            gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)

        ### END CODE HERE ###

        for i in gradients.keys():
                 gradients[i] = np.clip(gradients[i], -5, 5)

        parameters['Wax'] += -learning_rate * gradients['dWax']
        parameters['Waa'] += -learning_rate * gradients['dWaa']
        parameters['Wya'] += -learning_rate * gradients['dWya']
        parameters['b']  += -learning_rate * gradients['db']
        parameters['by']  += -learning_rate * gradients['dby']



        loss = loss * 0.999 + loss1 * 0.001

        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):

                # Sample indices and print them
                Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
                vocab_size = by.shape[0]
                n_a = Waa.shape[1]
                x = np.zeros((vocab_size,1))
                a_prev = np.zeros((n_a,1))
                indices = []
                idx = -1
                counter = 0
                newline_character = char_to_ix['\n']

                while (idx != newline_character and counter != 50):

                    a = np.tanh(np.matmul(Wax, x)+np.matmul(Waa, a_prev)+b)
                    z = np.matmul(Wya, a_prev)+by
                    y = softmax(z)

                    # for grading purposes
                    np.random.seed(counter+seed)

                    # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
                    idx = np.random.choice(list(char_to_ix.values()), p= y.ravel())

                    # Append the index to "indices"
                    indices.append(idx)

                    # Step 4: Overwrite the input character as the one corresponding to the sampled index.
                    x = np.zeros((vocab_size, 1))
                    x[idx] = 1

                    # Update "a_prev" to be "a"
                    a_prev = a

                    # for grading purposes
                    seed += 1
                    counter +=1


                if (counter == 50):
                    indices.append(char_to_ix['\n'])
                sampled_indices = indices

                txt = ''.join(ix_to_char[ix] for ix in sampled_indices)
                txt = txt[0].upper() + txt[1:]  # capitalize first character
                print ('%s' % (txt, ), end='')

                seed += 1  # To get the same result for grading purposed, increment the seed by one.

            print('\n')

    return parameters
parameters = model(data, ix_to_char, char_to_ix)
