import numpy as np



input_vector = np.array([1.2, 2.4, 3.6, 4.8, 6.0 , 999 , 76868 , 556453356])
input_weight = np.array([1.9, 3.8, 5.7, 7.6, 9.5 , 75675465 , 6868968967 , 5678578578])

bias_vector = np.array([0.0])


def sigmoid(x):
    return 1/(1+np.exp(-x))

def make_prediction(input_vector ,input_weight,bias_vector):
    layer_1 = np.dot(input_vector, input_weight)+bias_vector
    layer_2 = sigmoid(layer_1)
    return layer_2


prediction = make_prediction(input_vector,input_weight,bias_vector)
print(f"The Prediction is : {prediction}")




