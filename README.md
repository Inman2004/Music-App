import numpy as np

import skfuzzy as fuzz

from skfuzzy import control as ctrl

import matplotlib.pyplot as plt

# Step 1: Define input and output variables

temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')

fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

# Step 2: Define membership functions

temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 50])

temperature['comfortable'] = fuzz.trimf(temperature.universe, [0, 50, 100])

temperature['hot'] = fuzz.trimf(temperature.universe, [50, 100, 100])

fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])

fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [0, 50, 100])

fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

# Step 3: Define fuzzy rules

rule1 = ctrl.Rule(temperature['cold'], fan_speed['high'])

rule2 = ctrl.Rule(temperature['comfortable'], fan_speed['medium'])

rule3 = ctrl.Rule(temperature['hot'], fan_speed['low'])

# Step 4: Create control system

fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# Step 5: Create simulation and compute

fan_sim = ctrl.ControlSystemSimulation(fan_ctrl)

input_temperature = 75

fan_sim.input['temperature'] = input_temperature

fan_sim.compute()

output_fan_speed = fan_sim.output['fan_speed']

# Step 6: Display results

print(f"Input Temperature: {input_temperature}")

print(f"Output Fan Speed: {output_fan_speed:.2f}")

# Plot membership functions and result

temperature.view()

fan_speed.view()

plt.show()

Ex 2

import numpy as np
class DiscretePerceptron:
 def __init__(self, input_size):
 self.weights = np.random.rand(input_size)
 self.bias = np.random.rand()
 
 def predict(self, inputs):
 weighted_sum = np.dot(inputs, self.weights) + self.bias
 return 1 if weighted_sum > 0 else 0
 
 def train(self, inputs, targets, learning_rate=0.1, epochs=100):
 for epoch in range(epochs):
 for i in range(len(inputs)):
 prediction = self.predict(inputs[i])
 error = targets[i] - prediction
 self.weights += learning_rate * error * inputs[i]
 self.bias += learning_rate * error

 def main():
 # Training data: XOR problem
 inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
 targets = np.array([0, 1, 1, 0])
 
 perceptron = DiscretePerceptron(input_size=2)
 perceptron.train(inputs, targets, learning_rate=0.1, epochs=100)
 
 # Testing
 test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
 for data in test_data:
 prediction = perceptron.predict(data)
 print(f"Input: {data}, Prediction: {prediction}")
if __name__ == "__main__":
 main()
