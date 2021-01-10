# Simple Multilayer Perceptron with Backpropagation trained for XOR


## Introduction
First of all, what is a multilayer perceptron? In a few words, a MLP is a type of feedforward artificial neural network [1]. This network consists of at least three layers of neurons:
* input layer
* hidden layer
* output layer

Of course, one can have multiple hidden layers. For my example, I have created a network structure as below:

![Structure of the network](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/neural_network_structure.png)

One neuron has the following structure [1]:

![Structure of one neuron](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/structure-of-a-single-node-of-an-NN-a-neuron.ppm.png)
## Idea

After reading material on how to build up a neural network and what would be the best for my case, I came up with the following structure:

The XOR dataset would be generated by the application using 2^n number of inputs. Basically we could generate any number of inputs for the input layer which would cover all the combinations of inputs for XOR, as following:

For n = 2, we would have the following dataset created for the training:

```python
{(0, 0): [0], 
 (0, 1): [1], 
 (1, 0): [1], 
 (1, 1): [0]}
```

The righthand side represents the XOR result between the two values, set beforehand and fed to the network for training. Since this is a supervised learning, we neet to let the network know beforehand what the expected result is, thus we are also giving the righthand side to the training. After the training, for each entry in the dataset, we should have a close enough estimation of the output. For an example run, we have the following result:

Network at iteration 4435 with all squared mean error of 0.0009999758016979844:

```python
{(0, 0): [0.028281311827909387],
 (0, 1): [0.9670412803392583],
 (1, 0): [0.9670108883329275],
 (1, 1): [0.03193226003479527]}
```

More details of the implementation of the algorithm and the logic behind it will be found at the Details section.

## Concept

A neural network is formed of multiple layers of neurons. Each neuron has connections to neurons from the neighbour layers. Each connection has a weight, which is randomly set at the beginning of the training; during the training, if we use backwards propagation, the weights will be adjusted accordingly based on the delta value of each neuron and the derivative of the activation function. 
In order to get better results for our network, we use the backwards propagation technique:
* forward propagate
* back propagate
* adjust weights and bias

### Forward propagation

Feedforward neural networks are a type of networks in which the connections between the nodes do not form a cycle. Therefore, the information is moving in one direction only through the network: from the input to the hidden layers, which will feed the result in the output layer. In order to create a multilayer perceptron, I created a feedforward artificial neutal network. One difference between my implementation and the classic ones is that instead of treating the input as neurons, I merely treated the input as a list. I have chosen to do so because the inputs should not change when we adjust the weights and biases. So the neural network actually looks something like this instead:

TODO INSERT PICTURE OF NO INPUT LAYER NN

#### What does forward propagation do?

Forward propagation through the network means that each neuron will calculate its output as: 
* Calculating the sum between between each connected neuron's input, multiplied with the weight of the connection;
* Adding the bias of the current neuron after generating the above sum, as:

![Structure of one neuron](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/structure-of-a-single-node-of-an-NN-a-neuron.ppm.png)
* Apply a chosen activation function to the resulting sum.

The result after applying the activation function will represent the output for the current neuron, which will be passed to the neurons in the following layer which are connected to the current one. 

I have implemented this at neuron level, as follows:

```python
def forward_propagate(self, input):
        """
        Forward propagate.

        The function forward propagates through the neural network by
        constructing the sum between each input and its weight, adding the bias
        at the end. The result represents the output for this neuron which will
        be paseed to the next layer.
        """
        if len(input) != len(self.__weights):
            raise Exception("The number of input is different than the number"
                            "of weights in {}".format(self))

        sum = 0
        for i, w in zip(input, self.__weights):
            sum += i * w

        sum += self.__bias

        output = self.__activation_function(sum)
        self.__output = output

        return output
```
After the forward propagation phase, we will have an output for each layer based on the formula above. But this is not enough, since the weights are chosen at random and probably the network is not adjusted enough in order to chose with an acceptable accuracy which is the result for solving XOR problems. Therefore, we can use the back propagation technique which tells the network by how much to adjust its weights in order to get better, more accurate results. 

More information about forward propagation can be found at [2], [3] and [4].

### Back propagate

In order to fit our neural network, we need to adjust its weights using the back propagation method, detailed as:

In fitting a neural network, backpropagation computes the gradient of the loss function with respect to the weights of the network for a single input–output example, and does so efficiently, unlike a naive direct computation of the gradient with respect to each weight individually. This efficiency makes it feasible to use gradient methods for training multilayer networks, updating weights to minimize loss; gradient descent, or variants such as stochastic gradient descent, are commonly used. The backpropagation algorithm works by computing the gradient of the loss function with respect to each weight by the chain rule, computing the gradient one layer at a time, iterating backward from the last layer to avoid redundant calculations of intermediate terms in the chain rule; this is an example of dynamic programming [5].

### What does back propagation do?

In machine learning, the delta rule is a gradient descent learning rule for updating the weights of the inputs to artificial neurons in a single-layer neural network [9]. More information about this can be found [6], [7] and [8]. However, we have two casesfor calculating the delta between the output layer and the expected results and calculating the delta between two layer of the perceptron (in my case, the hidden layer and the output layer).

#### Delta calculation between output layer and expected results

In order to generate the delta values for the output neurons, we need to get the difference of error between each neuron's output and the expected output. This difference between errors is then multiplied with the derived output of the neuron (using the derivative of the activation function), which generates the delta.

I have implemented this at layer level since this is an action which takes part between the output layer and the expected results:

```python
def backward_propagate_output(self, expected_outputs):
        """Backpropagate the error between the output layer and expected.

        This is a special case because we calculate the error between the
        output layer and the expected results, not the error difference between
        one layer and another.
        """
        if len(self.__neurons) != len(expected_outputs):
            raise Exception("The number of neurons in the layer versus the"
                            "number ofexpected outputs are different."
                            "Cannot backpropagate.")

        for neuron, expected_output in zip(self.__neurons, expected_outputs):
            error = expected_output - neuron.get_output()
            delta = error * neuron.get_derived_output()
            neuron.set_delta(delta)
```

#### Delta calculation between different layers
The difference between calculating the delta of the output layer is that for hidden layers, we actually need to get the weighted inputs from each neuron's connections and add them to the error. After adding all the multiplications between the previous weights and inputs which were used in feed forward to generate the current neuron's output, we multiply them with the derived output and get delta.

I have also implemented this at layer level since this is too an action between two layers, rather than a neuron level one.

```python
def backward_propagate_hidden(self, previous_layer):
        """Backpropagate the delta error between two layers."""
        # Calculate the error generated by all previous neurons connected to
        # the ones from the previous layer. Generate their deltas based on the
        # weighted sum beween each previous neuron with its weight connected to
        # the current one. The sum of errors is multiplied with the derivative
        # of the activation function used to get the results.
        for i, neuron in enumerate(self.__neurons):
            previous_layer_neurons = previous_layer.get_neurons()
            error = 0

            for pln in previous_layer_neurons:
                pln_error = pln.get_delta() * pln.get_weight(index=i)
                error += pln_error

            delta = error * neuron.get_derived_output()

            neuron.set_delta(delta)
```
### Adjust weights

After generating the deltas for each neuron, we parse the network and adjust the weights according to the previously generated delta, as follows:

```python
def adjust(self, learning_rate, input_neurons_list):
        """
        Adjust the weights after backpropagating.

        After the backward propagation of error is done, the weights of all the
        neurons are adjusted according to the following formula:

        adjusted_weight = weight + learning_rate * self.__delta * output

        The bias will also be adjusted.
        """
        adjusted_weights = []

        if len(self.__weights) != len(input_neurons_list):
            raise Exception("Weights and input neurons list are different"
                            "{} vs {}. Cannot adjust weights."
                            .format(self.__weights, input_neurons_list))

        for weight, input_neuron in zip(self.__weights, input_neurons_list):
            output = input_neuron.get_output()
            adjusted_weight = weight + learning_rate * self.__delta * output
            adjusted_weights.append(adjusted_weight)

        self.__weights = adjusted_weights
        self.__bias = self.__bias + learning_rate * self.__delta
````
### Repeat 

In order to improve the accuracy of our network, we then need to repreat these steps and keep tweaking the weights until we find an acceptable mean squared error for the network, as follows:

```python
def squared_mean_error(self, expected_output):
        """Compute the squared mean error between the output and expected.

        The squared mean error is calculated at the output layer between the
        results generated by the neuron network and the expected results. More
        info at:https://www.freecodecamp.org/news/
        machine-learning-mean-squared-error-regression-line-c7dde9a26b93/
        """
        squared_mean_error = 0

        for neuron, eo in zip(self.__neurons, expected_output):
            error = eo - neuron.get_output()
            squared_mean_error += error * error

        squared_mean_error = squared_mean_error / len(self.__neurons)
        return squared_mean_error
```
## Implementation Details

I have tried to keep the implementation as simple and as structured as possible. Of course, there is room for improvement. For now I have the following project structure:

![Simple Multilayer Perceptron project structure.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/simple_mlp_folder_structure.png)

### Network

![Network class structure.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/network_class.png)

### Layer

![Layer class structure.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/layer_class.png)

### Neuron

![Neuron class structure.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/neuron_class.png)

### Train

![Train class structure.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/train_class.png)

### Relationships

![Classes relationship.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/classes_architecture_simple.png)

## Experiments

After I have put my network on place, I have started experimenting with different learning rates, network structures and activation functions. 

### Dataset

I have generated the XOR dataset using the following function, called for input size 2:

```python
def generate_xor_dataset(self, input_size):
        """
        Generate an XOR dataset for the training.

        This generates a 2^(input_size) data set of xor combinations.
        """
        xor_input_combinations = list(itertools.product([0, 1],
                                      repeat=input_size))
        self.__dataset = {}

        # set the output for supervised learning
        for combination in xor_input_combinations:
            self.__dataset[combination] = \
                [functools.reduce(lambda x, y: x ^ y, combination)]
```
### Network generation

I have set the network based on the number of inputs, network structure, the random generated weights, activation functions and derivatives for each neuron. These will be passed to each layer, which will pass them to each neuron. After the setup network, we will have our network ready up and ready to be trained. This generated a network as the one shown in the first figure, with 2 neurons for the hidden layer and one for the output layer.

```python
def setup_network(self,
                      input_count,
                      network_layout,
                      activation_functions,
                      activation_functions_derivations):
        """
        Set up the network for the training.

        In order to generate the network with random weights, we need to know
        the network layout and the number of inputs for the layout.
        """
        network_weights = self.generate_random_network_weights(input_count,
                                                               network_layout)

        network_activation_functions = activation_functions
        network_activation_functions_derivatives = \
            activation_functions_derivations

        self.__network = \
            n.Network(network_weights,
                      network_activation_functions,
                      network_activation_functions_derivatives)

        return self.__network
```
### Train parameters

After generating the dataset and setting up the network, we still have to choose the learning rate for the gradient descent, as well as an acceptable mean squared error such that the training stops after it gets to this value (this was just prefference, since I did not want to stop it by hand each time and the iterations were too fast for me to follow on the command line interface). After experimenting with different learning rates, I have chosen learning rate = 0.5, which seemed too much. But I ended up with good results in approximately 4000-6000 iterations in most of the cases (until I reached the set acceptable squared mean error at 0.001). I have saved some of these experiments in experiments_X.txt (where X is a number) files which are attached to the project. 

### Experiment 1


| Learning Rate | Acceptable Mean Squared Error | Network Layout | Activation function |
| ------------- | ----------------------------- | -------------- | ------------------- |
| 0.001		| 0.001				| [2, 1]	 | Sigmoid |

#### Result

| Input | Expected output | Generated output | Iterations |
|----- | --------------- | ----------------| ----------|
| (0, 0) | 0 | [0.5014284968178055] | Manually stopped at 721850 |
| (0, 1) | 1 | [0.49135766308898915] | Manually stopped at 721850 | 
| (1, 0) | 1 | [0.49857150318219534] | Manually stopped at 721850 | 
| (1, 1) | 0 | [0.5086423369110108] | Manually stopped at 721850 | 

#### Squared mean errors: 0.25507378215962895.

At this point I actually thought I have implemented something completely wrong since it is ending up in generating almost 0.5 when the expected result should be as close to 0 or as close to 1. But it was just the learning rate too slow, and I think it got stuck in a local minimum and was "lucky" enough to generate these confusing results.

#### Weights and bias before training:

![Experiment 1 - weights and biases before training.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/Experiments_0_diagram_before.png)

#### Weights and bias after training:

![Experiment 1 - weights and biases after training.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/Experiments_0_diagram_after.png)

### Experiment 2


| Learning Rate | Acceptable Mean Squared Error | Network Layout | Activation function |
| ------------- | ----------------------------- | -------------- | ------------------- |
| 0.05    | 0.001                         | [2, 1]         | Sigmoid |

#### Result

| Input | Expected output | Generated output | Iterations |
|----- | --------------- | ----------------| ----------|
| (0, 0) | 0 | [0.03433052150097361] | 40169 |
| (0, 1) | 1 | [0.9698091889594029]  | 40169 |
| (1, 0) | 1 | [0.9698128642022481] | 40169 |
| (1, 1) | 0 | [0.031599633874256135] | 40169 |

#### Squared mean errors: 0.0009999778359190655

#### Weights and bias before adjust

![Experiment 2 - weights and bias before training.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/Experiments_11_diagram_before.png)
#### Weights and bias after adjust

![Experiment 2 - weights and bias after training.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/Experiments_11_diagram_after.png)

### Experiment 3


| Learning Rate | Acceptable Mean Squared Error | Network Layout | Activation function |
| ------------- | ----------------------------- | -------------- | ------------------- |
| 0.5    | 0.001                         | [2, 1]         | Sigmoid |

#### Result

| Input | Expected output | Generated output | Iterations |
|----- | --------------- | ----------------| ----------|
| (0, 0) | 0 | [0.028281311827909387] | 4435 |
| (0, 1) | 1 | [0.9670412803392583]  | 4435 |
| (1, 0) | 1 | [0.9670108883329275] | 4435 |
| (1, 1) | 0 | [0.03193226003479527] | 4435 |

#### Squared mean errors: 0.0009999758016979844

#### Weights and bias before adjust

![Experiment 3 - weights and bias before training.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/Experiments_12_diagram_before.png)
#### Weights and bias after adjust

![Experiment 3 - weights and bias after training.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/Experiments_12_diagram_after.png)

### Experiment 4


| Learning Rate | Acceptable Mean Squared Error | Network Layout | Activation function |
| ------------- | ----------------------------- | -------------- | ------------------- |
| 0.2    | 0.0001                         | [2, 1]         | Sigmoid |

#### Result

| Input | Expected output | Generated output | Iterations |
|----- | --------------- | ----------------| ----------|
| (0, 0) | 0 | [0.009215194470002675] | 69454 |
| (0, 1) | 1 | [0.9895387006570503]  | 69454 |
| (1, 0) | 1 | [0.9895390356793257] | 69454 |
| (1, 1) | 0 | [0.009807392986486852] | 69454 |

#### Squared mean errors: 9.999942829450311e-05

#### Weights and bias before adjust

![Experiment 4 - weights and bias before training.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/Experiments_13_diagram_before.png)
#### Weights and bias after adjust

![Experiment 4 - weights and bias after training.](https://github.com/BabyCakes13/Simple-Multilayer-Perceptron/blob/master/pics/Experiments_13_diagram_after.png)

## Conclusions

The experience of building a neural network from grounds up, training it and getting to experiment with various little changes is a really fun and satisfying one. There is room for more improvements, as well as from the implementation point of view (such as treating the input layer as a layer of neurons rather than an array), as well as finding better structures for even more accuracy. 
Some interesting and valuable material which I found is linked in the Bibliography at [10], [11], 12[], [13] and [14], besides the already linked ones. 

## Bibgliography
1. https://stackoverflow.com/questions/26058022/neural-network-activation-function-vs-transfer-function/26059347
2. https://towardsdatascience.com/forward-propagation-in-neural-networks-simplified-math-and-code-version-bbcfef6f9250
3. https://en.wikipedia.org/wiki/Feedforward_neural_network
4. https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
5. Goodfellow, Bengio & Courville 2016, p. 214, "This table-filling strategy is sometimes called dynamic programming."
6. https://en.wikipedia.org/wiki/Delta_rule
7. https://medium.com/@neuralnets/delta-learning-rule-gradient-descent-neural-networks-f880c168a804
8. https://www.youtube.com/watch?v=Ilg3gGewQ5U
9. Russell, Ingrid. "The Delta Rule". University of Hartford. Archived from the original on 4 March 2016. Retrieved 5 November 2012.
10. https://www.youtube.com/watch?v=aircAruvnKk&t=2s (the whole series)
11. https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
12. https://becominghuman.ai/neural-network-xor-application-and-fundamentals-6b1d539941ed
13. https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e
14. https://towardsdatascience.com/how-do-artificial-neural-networks-learn-773e46399fc7
