import time

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

class Fbase():
    def __init__(self, R, K, alpha, beta, epochs):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs

    def plot_errors(self, errors, pic_loss=None, show=True):
        plt.figure(figsize=(10,6))
        plt.title('errors')
        plt.xlabel('epoch')
        plt.ylabel('mse')
        plt.plot(errors, label='mse')
        #plt.plot(self.val_losses, label='val_loss')
        plt.legend(loc='best')
        plt.grid() #which='both'
        if pic_loss: plt.savefig(self.pic_loss)
        if show: plt.show()
        plt.close()

class MF(Fbase):
    def __init__(self, R, K, alpha, beta, epochs):
        super().__init__(R, K, alpha, beta, epochs)
        '''
        MF(R, K=5, alpha=0.1, beta=0.01, epochs=20)
        Perform matrix factorization to predict empty entries in a matrix.
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        '''

    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K)) # rows, users matrix
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K)) # cols, items

        self.b_u = np.zeros(self.num_users) #biases
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])                                # <--- !
        
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0                                                        # <--- !
        ] # training samples
        
        errors = []
        for i in range(self.epochs):         # stochastic gradient descent # epochs
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            errors.append(mse)
            if (i+1) % 50 == 0:
                print("it: %d ; error = %.4f" % (i+1, mse))

        return errors

    def mse(self):
        '''compute the total mean square error'''
        xs, ys = self.R.nonzero()
        R_hat = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - R_hat[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        '''stochastic graident descent'''
        for i, j, r in self.samples:
            r_hat = self.get_rating(i, j) # prediction and error
            e = (r - r_hat)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i]) # update biases
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            P_i = self.P[i, :][:] # copy of row of P for update, but use older values for update on Q
            
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:]) # update user/item matrices
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        '''predicted rating of user i and item j'''
        return self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
    
    def full_matrix(self):
        '''full matrix using the biases, P and Q '''
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

def main():
    R = np.array([
      [0.0, 4.5, 2.0, 0.0],
      [4.0, 0.0, 3.5, 0.0],
      [0.0, 5.0, 0.0, 2.0],
      [0.0, 3.5, 4.5, 1.0]
    ])

    #alpha : learning rate, beta : regularization parameter, _lambda : reqularization
    mf = MF(R, K=5, alpha=0.1, beta=0.01, epochs=20)
    errors = mf.train()
    mtx = mf.full_matrix()
    #mtx = np.rint(mtx)
    print('R:\n', R)
    print('R_hat:\n', mtx)
    mf.plot_errors(errors)
    
if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Executed in {end - start:0.5f}s')

'''
R ~ P x Q^T = R_hat
r_hat_ij = pi^T*gj = sum_K(pik*qjk)
e^2 = (r_ij - r_hat_ij)^2

reqularize
e^2 = (r_ij - r_hat_ij)^2 + L2()

'''