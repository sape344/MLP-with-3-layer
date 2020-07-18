from sklearn.metrics import accuracy_score
import numpy as np
class Neural_Network():   
    def __init__(self,X_train,y_train,Number_Dense1,Number_Dense2,learning_rate,batch):
        self.X_train=X_train
        self.y_train=y_train
        self.Number_Dense1=Number_Dense1 #76
        self.Number_Dense2=Number_Dense2 #76
        self.parameters={}
        self.Adam_parameters={}
        self.catch={}
        self.grads={}
        self.learning_rate=learning_rate
        self.batch=batch
        

        

    
    
 
    def initialize_parameters_and_layer_sizes_NN(self):
        parameters={"weight1":np.random.randn(self.Number_Dense1,self.X_train.shape[0])*0.01,
                    "bias1":np.zeros((self.Number_Dense1,1)),
                    "weight2":np.random.randn(self.Number_Dense2,self.Number_Dense1)*0.01,
                    "bias2":np.zeros((self.Number_Dense2,1)),
                    "weight3":np.random.randn(self.y_train.shape[0],self.Number_Dense2)*0.01,
                    "bias3":np.zeros((self.y_train.shape[0],1))       
        }
        self.parameters=parameters
        
    def initialize_parameters_Adam_NN(self):        
        parameters={"weight1_vdw":np.zeros((self.Number_Dense1,self.X_train.shape[0])),
                    "bias1_vdw":np.zeros((self.Number_Dense1,1)),
                    "weight2_vdw":np.zeros((self.Number_Dense2,self.Number_Dense1)),
                    "bias2_vdw":np.zeros((self.Number_Dense2,1)),
                    "weight3_vdw":np.zeros((self.y_train.shape[0],self.Number_Dense2)),
                    "bias3_vdw":np.zeros((self.y_train.shape[0],1)),
                    "weight1_sdw":np.zeros((self.Number_Dense1,self.X_train.shape[0])),
                    "bias1_sdw":np.zeros((self.Number_Dense1,1)),
                    "weight2_sdw":np.zeros((self.Number_Dense2,self.Number_Dense1)),
                    "bias2_sdw":np.zeros((self.Number_Dense2,1)),
                    "weight3_sdw":np.zeros((self.y_train.shape[0],self.Number_Dense2)),
                    "bias3_sdw":np.zeros((self.y_train.shape[0],1))       
        }
        self.Adam_parameters=parameters
        
        
        
        
        
    def Adam_optimizer(self,vdw,epoch,sdw,DW,w,learningRate,epsilon = np.array([pow(10, -8)])):                  
        vdw = 0.9 * vdw + (0.1) * DW
        sdw = 0.999 * sdw + (0.001) * pow(DW, 2)
        vdw_corrected = vdw / (1-pow(0.9, epoch+1))
        sdw_corrected = sdw / (1-pow(0.999,epoch+1))
        w = w + learningRate * (vdw_corrected / (np.sqrt(sdw_corrected) + epsilon)) #- değiştir
        return w,vdw,sdw

        
    def compute_cost(self,AL, Y): 
        m = Y.shape[1]
        logprobs = np.multiply(np.log(AL),Y) +  np.multiply(np.log(1-AL), (1-Y))
        cost = -1/m*np.sum(logprobs)
        cost = np.squeeze(cost)
        return cost

        
    def Relu(self,x):
        for i in range(x.shape[0]):
            for i1 in range(x.shape[1]):
                if x[i,i1]<0:
                    x[i,i1]=0

        return x
    
    def ReluPrime(self,x):
        for i in range(x.shape[0]):
            for i1 in range(x.shape[1]):
                if(x[i,i1]<0):
                     x[i,i1]*=0
                else:
                    x[i,i1]*=1
                
        return x
    def stable_softmax(self,X):        
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)
    def der_tanh(self,X):       
        return 1-(np.tanh(X)**2)
    
    def cross_entropy_loss(self,Y,A3):
        E=np.zeros(Y.shape)
        
        for i in range(Y.shape[0]):
            for i1 in range(Y.shape[1]):
                if int(Y[i,i1])==1:
                    E[i,i1]=-np.log(A3[i,i1])
                else:
                    E[i,i1]=-np.log(1-A3[i,i1])
        
    
        return np.sum(E)

    def forward_propagation_NN(self,X_train):
        Z1=np.dot(self.parameters["weight1"],X_train)+self.parameters["bias1"]
        A1=np.tanh(Z1)
        Z2=np.dot(self.parameters["weight2"],A1)+self.parameters["bias2"]
        A2=np.tanh(Z2)
        Z3=np.dot(self.parameters["weight3"],A2)+self.parameters["bias3"]       
        A3=self.sigmoid(Z3)
    

        self.cache={
            "A1":A1,
            "A2":A2,
            "A3":A3,
            "Z1":Z1,
            "Z2":Z2,
            "Z3":Z3}
        return A3
    
    def sigmoid(self,s):
        return 1/(1+np.exp(-s))
        
    def sigmoidPrime(self,s):
        return s*(1-s)
    



    
 

        
    def backward_propagation_NN(self,X,Y):


        error=self.compute_cost(self.cache["A3"],Y)
        DW3_prime=(self.cache["A3"]-Y)*self.sigmoidPrime(self.cache["Z3"]) #DZ
        DW3=np.dot(DW3_prime,self.cache["A2"].T)/self.cache["A3"].shape[1]
        db3=np.sum(DW3_prime,axis=1,keepdims=True)/self.cache["A3"].shape[1]



        
        DW2_prime=(np.dot(self.parameters["weight3"].T,DW3_prime)*self.der_tanh(self.cache["Z2"]))#76*1000 =
        DW2=np.dot(DW2_prime,self.cache["A1"].T)/self.cache["A2"].shape[1] #DW2=76*76=    76*1000,1000*76
        db2=np.sum(DW2_prime,axis=1,keepdims=True)/self.cache["A2"].shape[1]

        
        
        DW1_prime=np.dot(self.parameters["weight2"].T,DW2_prime)*self.der_tanh(self.cache["Z1"])
        DW1=np.dot(DW1_prime,X.T)/self.cache["A1"].shape[1]
        db1=np.sum(DW1_prime,axis=1,keepdims=True)/self.cache["A1"].shape[1]



        
        self.grads={ "dweight1":DW1,
              "dbias1":db1,
              "dweight2":DW2,
              "dbias2":db2,
              "dweight3":DW3,
              "dbias3":db3}
        return error
        
            
    def softmax(self,X):
        exps = np.exp(X)
        return exps / np.sum(exps)

    def update_parameters_NN(self):
        parameter={"weight1":self.parameters["weight1"]-(self.learning_rate*self.grads["dweight1"]),
                "weight2":self.parameters["weight2"]-(self.learning_rate*self.grads["dweight2"]),
                   "weight3":self.parameters["weight3"]-(self.learning_rate*self.grads["dweight3"]),
                "bias1":self.parameters["bias1"]-(self.learning_rate*self.grads["dbias1"]),
                   "bias3":self.parameters["bias3"]-(self.learning_rate*self.grads["dbias3"]),
                "bias2":self.parameters["bias2"]-(self.learning_rate*self.grads["dbias2"])}
        self.parameters=parameter
        
    def Adam_update_parameters_NN(self,epoch):                      
        self.parameters["weight1"],self.Adam_parameters["weight1_vdw"],self.Adam_parameters["weight1_sdw"]=self.Adam_optimizer(self.Adam_parameters["weight1_vdw"],epoch,self.Adam_parameters["weight1_sdw"],self.grads["dweight1"],self.parameters["weight1"],self.learning_rate,np.array([pow(10, -8)]))
        self.parameters["bias1"],self.Adam_parameters["bias1_vdw"],self.Adam_parameters["bias1_sdw"]=self.Adam_optimizer(self.Adam_parameters["bias1_vdw"],epoch,self.Adam_parameters["bias1_sdw"],self.grads["dbias1"],self.parameters["bias1"],self.learning_rate,np.array([pow(10, -8)]))
        self.parameters["weight2"],self.Adam_parameters["weight2_vdw"],self.Adam_parameters["weight2_sdw"]=self.Adam_optimizer(self.Adam_parameters["weight2_vdw"],epoch,self.Adam_parameters["weight2_sdw"],self.grads["dweight2"],self.parameters["weight2"],self.learning_rate,np.array([pow(10, -8)]))
        self.parameters["bias2"],self.Adam_parameters["bias2_vdw"],self.Adam_parameters["bias2_sdw"]=self.Adam_optimizer(self.Adam_parameters["bias2_vdw"],epoch,self.Adam_parameters["bias2_sdw"],self.grads["dbias2"],self.parameters["bias2"],self.learning_rate,np.array([pow(10, -8)]))
        self.parameters["weight3"],self.Adam_parameters["weight3_vdw"],self.Adam_parameters["weight3_sdw"]=self.Adam_optimizer(self.Adam_parameters["weight3_vdw"],epoch,self.Adam_parameters["weight3_sdw"],self.grads["dweight3"],self.parameters["weight3"],self.learning_rate,np.array([pow(10, -8)]))
        self.parameters["bias3"],self.Adam_parameters["bias3_vdw"],self.Adam_parameters["bias3_sdw"]=self.Adam_optimizer(self.Adam_parameters["bias3_vdw"],epoch,self.Adam_parameters["bias3_sdw"],self.grads["dbias3"],self.parameters["bias3"],self.learning_rate,np.array([pow(10, -8)]))
                      
        
        
    def prediction_NN(self,X_test):
        
        A2=self.forward_propagation_NN(X_test)
        print(A2.shape)
        Y_prediction=np.zeros((self.batch,5))
        for i1 in range(5):
            for i in range(self.batch):
                if A2[i,i1]<=0.5:
                    Y_prediction[i1,i]=0
                else:
                    Y_prediction[i1,i]=1
        return Y_prediction
        
    def train(self,X,y,epoch,X_validation,y_validation):
        
        self.initialize_parameters_and_layer_sizes_NN()
        self.initialize_parameters_Adam_NN()
        step_number=int(X.shape[1]/self.batch)
        i1=0
        for i1 in range(epoch):
            i=0
            if i1>0:
                print("Epoch number: "+str(i1)+"/"+str(epoch)+ "step_number: "+str(i)+"/"+str(step_number),"cost: ",error,"accuracy: ",accuracy_score(NN.forward_propagation_NN(X_test).argmax(axis=0), y_test.argmax(axis=0), normalize=True))

            for i in range(step_number):
                self.forward_propagation_NN((X[:,i*self.batch:(i+1)*self.batch]))
                error=self.backward_propagation_NN(X[:,i*self.batch:(i+1)*self.batch],y[:,i*self.batch:(i+1)*self.batch])
                self.Adam_update_parameters_NN(i1)
                
#Example

NN= Neural_Network(X_train,y_train,76,76,0.0001,1000) #v8
NN.train(X_train,y_train,10000,X_test,y_test)

