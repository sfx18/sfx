import numpy as np


class PartyNN():
    def __init__(self, learning_rate):
        self.weight_1 = np.random.normal(0.0, 1, (2, 4))
        self.weight_2 = np.random.normal(0.0, 1, (4, 1))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        input_1 = inputs.dot(self.weight_1)
        output_1 = np.array([self.sigmoid(x) for x in input_1])

        input_2 = output_1.dot(self.weight_2)
        output_2 = np.array([self.sigmoid(x) for x in input_2])

        return output_2

    def train(self, inputs, excepted):
        input_1 = inputs.dot(self.weight_1)
        output_1 = np.array([self.sigmoid(x) for x in input_1])

        input_2 = output_1.dot(self.weight_2)
        output_2 = np.array([self.sigmoid(x) for x in input_2])

        error = np.array([output_2[0] - excepted])
        dx = output_2[0] * (1 - output_2[0])
        weights_delta = error * dx
        self.weight_2 = self.weight_2 - output_1.reshape(1, len(output_1)).T * weights_delta * self.learning_rate

        error = self.weight_2 * weights_delta
        dx = output_1.reshape(len(output_1), 1) * (1 - output_1.reshape(len(output_1), 1))
        weights_delta = error * dx
        self.weight_1 = self.weight_1 - inputs.reshape(len(inputs), 1).dot(weights_delta.T) * self.learning_rate

def MSE(y, Y):
    return np.mean((y-Y)**2)

train = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
    ]

epochs = 500
learning_rate = 0.7
N = PartyNN(learning_rate = learning_rate)

for e in range(epochs):
    left = []
    right = []
    for data, excepted in train:
        N.train(np.array(data), excepted)
        left.append(np.array(data))
        right.append(np.array(excepted))

    loss = MSE(N.predict(np.array(left)).T, np.array(right))
print(loss)

print("weights:")
print(N.weight_1)
print(N.weight_2)