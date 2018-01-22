import numpy as np
import matplotlib.pyplot as plt
#Load necessary libraries


data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
data_y = np.sin(data_x) + 0.1 * np.power(data_x, 2) + 0.5 * np.random.randn(100, 1)
data_x /= np.max(data_x)
#Generate our data
#print (data_x)

data_x = np.hstack((np.ones_like(data_x), data_x))
#Add intercept data and normalize
#print (data_x)
#plt.plot (data_x)

order = np.random.permutation(len(data_x))
portion = 20
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]
#Shuffle data and produce train and test sets


def get_gradient(w, x, y):
    #y_estimate = x.dot(w).flatten()
    y_estimate = np.dot(x,w).flatten()
    error = (y.flatten() - y_estimate)
    #print (error)
    #print (len(error))
    #print ("this is the length of x " ,len(x))
    mse = (1.0/len(x))*np.sum(np.power(error, 2))
    #print (x)
    gradient = -(1.0/len(x)) * error.dot(x)
    return gradient, mse
#Create gradient function


w = np.random.randn(2)
alpha = 0.5
tolerance = 1e-5

# Perform Gradient Descent
iterations = 1
while True:
    gradient, error = get_gradient(w, train_x, train_y)
    new_w = w - alpha * gradient

    # Stopping Condition
    if np.sum(abs(new_w - w)) < tolerance:
        print ("Converged.")
        break

    # Print error every 50 iterations
    if iterations % 100 == 0:
        print ("Iteration: %d - Error: %.4f" %(iterations, error))

    iterations += 1
    w = new_w

print ("w =",w)
print ("Test Cost =", get_gradient(w, test_x, test_y)[1])
# Iteration: 100 - Error: 1.3025
#Iteration: 200 - Error: 1.2638
#Iteration: 300 - Error: 1.2636
#Iteration: 400 - Error: 1.2636
#Converged.

#w = [ -2.34360233  11.24503286]
#Test Cost = 2.13820437614
#Perform gradient descent to learn model

plt.plot(data_x[:,1], data_x.dot(w), c='g', label='Model')
plt.scatter(train_x[:,1], train_y, c='b', label='Train Set')
plt.scatter(test_x[:,1], test_y, c='r', label='Test Set')
plt.grid()
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
#Plot the model obtained

w1 = np.linspace(-w[1]*3, w[1]*3, 300)
w0 = np.linspace(-w[0]*3, w[0]*3, 300)
J_vals = np.zeros(shape=(w1.size, w0.size))

for t1, element in enumerate(w1):
    for t2, element2 in enumerate(w0):
        wT = [0, 0]
        wT[1] = element
        wT[0] = element2
        J_vals[t1, t2] = get_gradient(wT, train_x, train_y)[1]

plt.scatter(w[0], w[1], marker='*', color='r', s=40, label='Solution Found')
CS = plt.contour(w0, w1, J_vals, np.logspace(-10,10,50), label='Cost Function')
plt.clabel(CS, inline=1, fontsize=10)
plt.title("Contour Plot of Cost Function")
plt.xlabel("w0")
plt.ylabel("w1")
plt.legend(loc='best')
plt.show()
