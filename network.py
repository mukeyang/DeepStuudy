import  random
import  numpy as np
"""
implement a basic bp network with SGD
"""


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class network(object):
    def __init__(self,sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x ,y in zip(sizes[:-1],sizes[1:])]
    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        training_data = list(training_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.updata_mini_batch(mini_batch,eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test));
            else:
                print("Epoch {} complete".format(j))


    def updata_mini_batch(self, mini_batch, eta):
        n_b=[np.zeros(b.shape) for b in self.biases]
        n_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_n_b,delta_n_w=self.backprop(x,y)
            n_b=[nb+dnb for  nb,dnb in zip(n_b,delta_n_b)]
            n_w=[nw+dnw for nw ,dnw in zip(n_w,delta_n_w)]
        self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,n_w)]
        self.biases=[b-(eta/len(mini_batch)) *nb for b,nb in zip(self.biases,n_b)]
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def backprop(self, x, y):
        ub=[np.zeros(b.shape) for b in self.biases]
        uw=[np.zeros(w.shape) for  w in self.weights]
        #feedforward
        activition=x
        activitions=[x]
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activition)+b
            zs.append(z)
            activition=sigmoid(z)
            activitions.append(activition)
        #backward prop
        delta=self.cost_derivative(activition[-1],y)*sigmoid_prime(zs[-1])
        ub[-1]=delta
        uw[-1]=np.dot(delta,activitions[-2].T)
        for l in range(2,self.num_layers):
            z=zs[-l]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].T,delta)*sp
            ub[-l]=delta
            uw[-l]=np.dot(delta,activitions[-l-1].T)
        return ub,uw
    def cost_derivative(self,out_put_activitions,y):
        return  out_put_activitions-y





