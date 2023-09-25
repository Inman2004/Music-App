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

Run the a
