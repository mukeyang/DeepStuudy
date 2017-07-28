import load
training_data, validation_data, test_data = load.load_data_wrapper()
training_data = list(training_data)
import nw
import  network


net = nw.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)