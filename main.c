#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

/**
 * Holds Weights and Biases for the Neural Net
 */
typedef struct {
    double W1[2][4]; //Weight from input i to hidden j
    double b1[4]; // Biases for hidden layer
    double W2[4]; // Hidden to Output weights
    double b2; //Output Bias
} Network;

/**
 * Computes the sigmoid of x
 * 
 * Used to 'activate' a neuron during forward pass
 * adds non-linearity
 * 
 * returns the sigmoid of x
 */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * Computes the derivative of s
 * 
 * s must be a value already 'activated' by the sigmoid
 * 
 * returns the derivative of s
 */
double sigmoid_derivative(double s) {
    return s * (1.0 - s);
}

/**
 * Computes & returns random number between -1 & 1
 * 
 * Used to assign random weights for initialisation of neural net
 */
double rand_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

/**
 * Performs a forward pass on the neural Network
 * 
 * The inputs x1, x2 are ran through the neural network to produce a final output prediction.
 *
 * Params
 * ------
 * Network* net: pointer to the neural network attributes
 * double x1, x2: XOR inputs
 * double h[4]: intermediate/dummy array for calculations, one for each hidden node
 * double* out: pointer to the output
 * 
 * does not return
 */
void forward_pass(Network* net, double x1, double x2, double h[4], double* out) {
    // Layer 1: input -> hidden
    for (int i=0; i<4; i++) {
        //dot product of weights & input + plus bias for each hidden node
        double z = x1 * net->W1[0][i] + x2 * net->W1[1][i] + net->b1[i];
        //apply the sigmoid
        h[i] = sigmoid(z);
    }

    // Layer 2: Hidden -> Output
    double z_out = 0.0;
    //for each hidden node add the output together multiplied by the weight to the output
    for (int i=0; i<4; i++) {
        z_out += h[i] * net->W2[i];
    }
    z_out += net->b2; //add the bias of the output node
    *out = sigmoid(z_out); //apply the sigmoid to the final output
}

/**
 * Performs backpropagation on the neural network
 * to change weights & biases based on the error between the output and target
 * 
 * Network* net: pointer to the neural network attributes
 * double x1, x2: XOR inputs
 * double h[4]: intermediate/dummy array for calculations, one for each hidden node
 * double out: the output value
 * double target: the expected/target output
 * double learning_rate: value which influences the learning rate usually between 0.1-1
 * 
 * does not return
 */
void backward(Network* net, double x1, double x2, double h[4], double out, double target, double learning_rate) {
    // 1. Output layer error and delta
    double error = target - out;
    double delta_out = error * sigmoid_derivative(out);

    // 2. Hidden layer deltas (error propagated backward through W2)
    double delta_h[4];
    for (int j = 0; j<4; j++) {
        delta_h[j] = delta_out * net->W2[j] * sigmoid_derivative(h[j]);
    }

    // 3. Update output layer weights + bias
    for (int i=0; i<4; i++) {
        net->W2[i] += learning_rate * delta_out * h[i];
    }
    net->b2 += learning_rate * delta_out;

    // 4. Update hidden layer weights + bias
    for (int i=0; i<4; i++) {
        net->W1[0][i] += learning_rate * delta_h[i] * x1;
        net->W1[1][i] += learning_rate * delta_h[i] * x2;
        net->b1[i] += learning_rate * delta_h[i];
    }
}

/**
 * Computes 1 step/epoch of training
 * 
 * Performs one forward pass and backpropagation
 */
void train_step(Network* net, double x1, double x2, double target, double learning_rate) {
    double h[4], out;
    forward_pass(net, x1, x2, h, &out);
    backward(net, x1, x2, h, out, target, learning_rate);
}

/**
 * Main Loop
 * 
 * Initialises inputs, targets & neural network.
 * Performs 10000 epochs of training for each input type.
 * Prints to the console the output of the neural net
 * for each XOR input compared to its expected output.
 */
int main(void) {
    //Initialise random, inputs and target outputs
    srand(time(NULL));
    double inputs[4][2] = {{0,0},
                        {0,1},
                        {1,0},
                        {1,1}};
    double targets[4] = {0, 1, 1, 0};
    int epochs = 10000; //set epochs

    //intialise neural network
    Network net;
    for (int i=0; i<4; i++) {
        net.W1[0][i] = rand_weight();
        net.W1[1][i] = rand_weight();
        net.W2[i] = rand_weight();
        net.b1[i] = 0;
    }

    //Perform training for amount of epochs for each input
    for(int i=0; i<epochs; i++) {
        for(int j=0; j<4;j++) {
            train_step(&net, inputs[j][0], inputs[j][1], targets[j], 0.1);
        }
    }
    //Print results
    printf("\n--- Testing Trained Network ---\n");
    for (int i=0; i<4; i++) {
        double h[4], out;
        forward_pass(&net, inputs[i][0], inputs[i][1], h, &out);

        printf("Input: (%.0f, %.0f) -> Predicted: %.4f | Target: %.0f\n", inputs[i][0], inputs[i][1], out, targets[i]);
    }
    return 0;
}