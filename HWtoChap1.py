class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        output = 1 if weighted_sum >= self.threshold else 0
        return output

weights = [0.5, 0.5]
threshold = 1.0

neuron = McCullochPittsNeuron(weights, threshold)

input_val = [1, 0]

output = neuron.activate(input_val)

print("Output: ", output)