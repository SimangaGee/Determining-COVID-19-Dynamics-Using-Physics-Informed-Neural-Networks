import sys
sys.path.insert(0, 'eslearn/')
import tensorflow as tf                                                        
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

# Declaration of Global variables (X is time in Days) 
X =[]
S =[]
I =[]
R =[]
D =[]

#Setup of random functions

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, t, s, i , r ,d, layers):
        
        X = np.concatenate([t,s,i,r,d], 1)
                
        self.lb = t.min()
        self.ub = t.max()
                
        self.X = X
        
        self.t = X[:,0:1]
        self.s = X[:,1:2]
        self.i = X[:,2:3]
        self.r = X[:,3:4]
        self.d = X[:,4:5]
        
        
        self.layers = layers
        
        # Initialize the Neural Network
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # Initialize the equation parameters
        self.lambda_1 =  tf.Variable([0.0], dtype=tf.float32) # Beta
        self.lambda_3 =  tf.Variable([0.0], dtype=tf.float32) # Gamma
        self.lambda_5 =  tf.Variable([0.0], dtype=tf.float32) # Delta
        
        #Initialize the tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                      log_device_placement=True))
        
        # Initialize the equation Variables
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.s_tf = tf.placeholder(tf.float32, shape=[None, self.s.shape[1]])
        self.i_tf = tf.placeholder(tf.float32, shape=[None, self.i.shape[1]])
        self.r_tf = tf.placeholder(tf.float32, shape=[None, self.r.shape[1]])
        self.d_tf = tf.placeholder(tf.float32, shape=[None, self.d.shape[1]])
        
        
        self.s_pred, self.i_pred, self.r_pred,self.d_pred, self.f_s_pred, self.f_i_pred,self.f_r_pred, self.f_d_pred = self.net_NS(self.t_tf)
        
        #The loss function
        self.loss = tf.reduce_sum(tf.square(self.f_d_pred)) + \
                    tf.reduce_sum(tf.square(self.s_tf - self.s_pred)) + \
                    tf.reduce_sum(tf.square(self.i_tf - self.i_pred)) + \
                    tf.reduce_sum(tf.square(self.r_tf - self.r_pred)) + \
                    tf.reduce_sum(tf.square(self.d_tf - self.d_pred)) + \
                    tf.reduce_sum(tf.square(self.f_s_pred)) + \
                    tf.reduce_sum(tf.square(self.f_i_pred)) + \
                    tf.reduce_sum(tf.square(self.f_r_pred))
        
        # The optimizer function            
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    #The neural network
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
           
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev = xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, t):
        lambda_1 = self.lambda_1       
        lambda_3 = self.lambda_3
        lambda_5 = self.lambda_5
        
        psi_and_p = self.neural_net(tf.concat([t], 1), self.weights, self.biases)
        
        S = psi_and_p[:,0:1]
        I = psi_and_p[:,1:2]
        R = psi_and_p[:,2:3]
        D = psi_and_p[:,3:4]
        
        
        s_t = tf.gradients(S, t)[0]
        i_t = tf.gradients(I, t)[0]
        r_t = tf.gradients(R, t)[0]
        d_t = tf.gradients(D, t)[0]
    

        #The Residual functions
       
        f_s = s_t +  lambda_1*(S*I) 
        f_i = i_t -  lambda_1*(S*I) + lambda_3*I + lambda_5*I
        f_r = r_t  - lambda_3*I 
        f_d = d_t - lambda_5*I
        
        return S, I, R, D, f_s, f_i, f_r, f_d
    
    def callback(self,loss,lambda_1, lambda_3, lambda_5):
        print('Loss: %.3e, l1: %.7f, l3: %.7f, l5: %.7f' % (loss,lambda_1, lambda_3,lambda_5))
      
    def train(self, nIter): 

        tf_dict = {self.t_tf: self.t,self.s_tf: self.s, self.i_tf: self.i,self.r_tf: self.r , self.d_tf: self.d}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_3_value = self.sess.run(self.lambda_3)
                lambda_5_value = self.sess.run(self.lambda_5)
                
                
                print('It: %d, Loss: %.3e, l1: %.7f, l3: %.7f, l5: %.7f, Time: %.2f' % 
                      (it, loss_value, lambda_1_value,lambda_3_value, lambda_5_value,elapsed))
                start_time = time.time()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss,self.lambda_1,self.lambda_3,self.lambda_5],
                                loss_callback = self.callback)
            
    
    def predict(self, t_star):
        
        tf_dict = {self.t_tf: t_star}
        
        s_star = self.sess.run(self.s_pred, tf_dict)
        i_star = self.sess.run(self.i_pred, tf_dict)
        r_star = self.sess.run(self.r_pred, tf_dict)
        d_star = self.sess.run(self.d_pred, tf_dict)
        return s_star, i_star, r_star, d_star

#The main fuction of the program

if __name__ == "__main__": 
     
    nu = 0.01/np.pi

    N = 100
    
    layers = [1 ,20 ,20 ,20 , 4]
   
    with open('julyData.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
 
        #Read data from the CSV file
        for row in csv_reader:
            X.append(row['day'])
            S.append(row['susceptible'])
            I.append(row['active'])
            D.append(row['death'])
            R.append(row['recovered'])
            
    # Find total size of data
    total = np.size(X)
    
    #Reshaping the arrays of Data
    X = np.reshape(X,(total,1))
    S = np.reshape(S,(total,1))
    I = np.reshape(I,(total,1)) 
    R = np.reshape(R,(total,1)) 
    D = np.reshape(D,(total,1))
    
    #Make the array of data float type
    XX = np.array(X,dtype=float)
    SS = np.array(S,dtype=float)
    II = np.array(I,dtype=float)
    RR = np.array(R,dtype=float)
    DD = np.array(D,dtype=float)
    
    
    #Non-dimentionalize the data
    XX = np.true_divide(XX, 1)
    SS = np.true_divide(SS, 1148000)
    II = np.true_divide(II, 1148000)
    RR = np.true_divide(RR, 1148000)
    DD = np.true_divide(DD, 1148000)
         
# Training Data    
    idx = np.random.choice(N,N, replace=False)
    
    t_train = XX[idx,:]
    S_train = SS[idx,:]
    I_train = II[idx,:]
    R_train = RR[idx,:]
    D_train = DD[idx,:]     
    
    model = PhysicsInformedNN(XX,SS,II,RR,DD,layers) 
    model.train(1000)
    
    #Obtaining test result  
    test = ['5','12','18','34','60','110','133','151','169','187','203','230','253','276','286','297','320','342','373','393','418','437','463','478','510','516']  
     
    
    tes = np.size(test) 
    test = np.reshape(test,(tes,1)) 
    tested = np.array(test,dtype=float)
    s_pred, i_pred, r_pred, d_pred = model.predict(tested)
    
   
    
   


# side by side
    plt.Figure()
    plt.title("susceptible")
    plt.plot(tested,s_pred,'b.',linewidth=0.8)   
    plt.plot(XX,SS,'r-',linewidth=0.8)  
    plt.legend(["Predicted values","Actual values"])
    plt.xlabel('time(days)')
    plt.ylabel('Population')
    plt.savefig('sirdsS.pdf')
    plt.close()

    plt.Figure()
    plt.title("Infected")
    plt.plot(tested,i_pred,'b.',linewidth=0.8)   
    plt.plot(XX,II,'r--',linewidth=0.8)  
    plt.legend(["Predicted values","Actual values"])
    plt.xlabel('time(days)')
    plt.ylabel('Population')
    plt.savefig('sirdsI.pdf')
    plt.close()
    
    plt.Figure()
    plt.title("Recovered")
    plt.plot(tested,r_pred,'b.',linewidth=0.8)   
    plt.plot(XX,RR,'r-',linewidth=0.8)  
    plt.legend(["Predicted values","Actual values"])
    plt.xlabel('time(days)')
    plt.ylabel('Population')
    plt.savefig('sirdsR.pdf')
    plt.close()
    
    plt.Figure()
    plt.title("Death")
    plt.plot(tested,d_pred,'b.',linewidth=0.8)   
    plt.plot(XX,DD,'r-',linewidth=0.8)  
    plt.legend(["Predicted values","Actual values"])
    plt.xlabel('time(days)')
    plt.ylabel('Population')
    plt.savefig('sirdsD.pdf') 
    plt.close()
