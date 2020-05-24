import numpy, random,
lr = 1 #learning rate
bias = 1 #value of bias
weights = [random.random() for w in range(3)]
#weights generated in a list (3 weights in total for 2 neurons and the bias)

def Perceptron(input1, input2, expected_output):
   perceptron_output = input1*weights[0]+input2*weights[1]+bias*weights[2]

   # Tutaj dopisz kod, ktory bedzie zwracał obliczoną funkcje aktywacji (Heaviside), dla zmiennej perceptron_output
   #...
   #...
   #...

   error = expected_output - perceptron_output
   weights[0] += error * input1 * lr
   weights[1] += error * input2 * lr
   weights[2] += error * bias * lr

for i in range(50):
    Perceptron(1, 1, 1)  # True or true
    Perceptron(1, 0, 1)  # True or false
    Perceptron(0, 1, 1)  # False or true
    Perceptron(0, 0, 0)  # False or false

x = 0 #tutaj podaj swoje dane do testowania 0 lub 1
y = 0 #tutaj podaj swoje dane do testowania 0 lub 1
outputP = x*weights[0] + y*weights[1] + bias*weights[2]
if outputP > 0 : #activation function
   outputP = 1
else :
   outputP = 0
print(x, "or", y, "is : ", outputP)
