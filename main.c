#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    double W1[2][2]; //Weight from input i to hidden j
    double b1[2]; // Biases for hidden layer
    double W2[2]; // Hidden to Output weights
    double b2; //Output Bias
} Network;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double s) {
    // s is sigmoid(x)
    return s * (1.0 - s);
}

double rand_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0; // range [-1, 1]
}

void forward_pass(Network* net, double x1, double x2, double h[2], double* out) {
    // Layer 1: input -> hidden
    for (int i=0; i<2; i++) {
        double z = x1 * net->W1[0][i] + x2 * net->W1[1][i] + net->b1[i];
        h[i] = sigmoid(z);
    }

    // Layer 2: Hidden -> Output
    double z_out = h[0] * net->W2[0] + h[1] * net->W2[1] + net->b2;
    *out = sigmoid(z_out);
}

void backward(Network* net, double x1, double x2, double h[2], double out, double target, double learning_rate) {
    // 1. Output layer error and delta
    double error = target - out;
    double delta_out = error * sigmoid_derivative(out);

    // 2. Hidden layer deltas (error propagated backward through W2)
    double delta_h[2];
    for (int j = 0; j<2; j++) {
        delta_h[j] = delta_out * net->W2[j] * sigmoid_derivative(h[j]);
    }

    // 3. Update output layer weights + bias
    for (int i=0; i<2; i++) {
        net->W2[i] += learning_rate * delta_out * h[i];
    }
    net->b2 += learning_rate * delta_out;

    // 4. Update hidden layer weights + bias
    for (int i=0; i<2; i++) {
        net->W1[0][i] += learning_rate * delta_h[i] * x1;
        net->W1[1][i] += learning_rate * delta_h[i] * x2;
        net->b1[i] += learning_rate * delta_h[i];
    }
}

void train_step(Network* net, double x1, double x2, double target, double learning_rate) {
    double h[2], out;
    forward_pass(net, x1, x2, h, &out);
    backward(net, x1, x2, h, out, target, learning_rate);
}

int main(void) {
    srand(time(NULL));
    double inputs[4][2] = {{0,0},
                        {0,1},
                        {1,0},
                        {1,1}};
    double targets[4] = {0, 1, 1, 0};
    int epochs = 10000;

    //intialise neural network
    Network net;
    for (int i=0; i<2; i++) {
        net.W1[0][i] = rand_weight();
        net.W1[1][i] = rand_weight();
        net.W2[i] = rand_weight();
        net.b1[i] = 0;
    }

    for(int i=0; i<epochs; i++) {
        for(int j=0; j<4;j++) {
            train_step(&net, inputs[j][0], inputs[j][1], targets[j], 0.1);
        }
    }
    printf("\n--- Testing Trained Network ---\n");
    for (int i=0; i<4; i++) {
        double h[2], out;
        forward_pass(&net, inputs[i][0], inputs[i][1], h, &out);

        printf("Input: (%.0f, %.0f) -> Predicted: %.4f | Target: %.0f\n", inputs[i][0], inputs[i][1], out, targets[i]);
    }
    return 0;
}