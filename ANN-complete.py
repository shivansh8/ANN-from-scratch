# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:28:21 2020

@author: SHIVANSH
"""

import numpy as np
import pandas as pd


np.random.seed(29)
df=pd.read_csv('mode_median_cleaned.csv')

def lkrelu(z):
      if z>=0:
          return z
      else:
          return z/100
def lkrelu_diff(z):
      if z>=0:
          return 1
      else:
          return 0.01

class NN: 
    def __init__(self):
        self.copy_weights=[]
        self.copy_bias=[]
        self.eeta=0.005
        self.epochs=200
    
      
    
    def sigmoid(self,x):
          
          z = 1/(1 + np.exp(-x))
          return z
    
    def backprop(self,weights,Y,Z,A,index,X,bias):
        a3=A[2]
        a2=A[1]
        a1=A[0]
        z1=Z[0]
        z2=Z[1]
        z3=Z[2]
        
        C=(a3[0]-Y[index])**2
        
        del_C_a3=[2*(a3[0]-Y[index])]
        del_C_a3=np.array(del_C_a3)
        
        del_C_a2=np.zeros(3)
        
        for k in range(3):
            
            del_C_a2[k]=weights[2][0][k]*(lkrelu_diff(a3[0]))*((del_C_a3[0]))
            
            
        
        del_C_a1=np.zeros(3)
        for k in range(3):
            for j in range(3):
                
                del_C_a1[k]+=(weights[1][k][j])*(lkrelu_diff(a2[j]))*(del_C_a2[j])
                
        
        w1=[[0 for i in range(9)] for j in range(3)]
        
        w2=[[0 for i in range(3)] for j in range(3)]
        w3=[0 for i in range(3)]
        
        del_C_weights=[w1,w2,w3]
        
        
        for k in range(3):
          
          del_C_weights[2][k]=a2[k]*(lkrelu_diff(a3[0]))*(del_C_a3[0]) 
          
        
        for k in range(3):
          for j in range(3):
              
              del_C_weights[1][k][j]=a1[k]*(lkrelu_diff(a2[j]))*del_C_a2[j] 
              
        for k in range(3):
            for j in range(9):
              
              del_C_weights[0][k][j]=X[j]*(lkrelu_diff(a1[k]))*del_C_a1[k] 
              
        del_C_b3=np.zeros(1)
        
        del_C_b3[0]=(2*(a3[0]-Y[index])*(lkrelu_diff(a3[0])))
        del_C_b2=np.zeros(3)
        for i in range(3):
          
          del_C_b2[i]=(del_C_a2[i]*(lkrelu_diff(a2[i])))
          
        del_C_b1=np.zeros(3)
        for j in range(3):
          
          del_C_b1[j]=(del_C_a1[j]*(lkrelu_diff(a1[j])))
          
        
        for i in range(3):
          for j in range(9):
                weights[0][i][j]-=(self.eeta*del_C_weights[0][i][j])
        for i in range(3):
             for j in range(3):
                  weights[1][i][j]-=(self.eeta*del_C_weights[1][i][j])
        
        for j in range(3):
                  weights[2][0][j]-=(self.eeta*del_C_weights[2][j])
        for i in range(3):
              bias[0][i]-=(self.eeta*del_C_b1[i])
              bias[1][i]-=(self.eeta*del_C_b2[i])
        bias[2][0]-=(self.eeta*del_C_b3[0])
        
    
    
        
    
            
    
    
    def foward(self,weights,X,bias):
            graph1=weights[0]
            graph2=weights[1]
            graph3=weights[2]
            
            z1=np.dot(graph1,X)
            
            a1=[]
            for i in range(len(z1)):
                z1[i]+=bias[0][i]
                
                a1.append(lkrelu(z1[i]))
            a1=np.array(a1)
            
            z2=np.dot(graph2,a1)
            a2=[]
            
            for i in range(len(z2)):
              z2[i]+=bias[1][i]
              
              a2.append(lkrelu(z2[i]))
            a2=np.array(a2)
            
            a3=[]
            z3=np.dot(graph3,a2)
            
            for i in range(len(z3)):
              z3[i]+=bias[2][i]
              a3.append(self.sigmoid(z3[i]))
              
            a3=np.array(a3)
            
            
            return [[z1,z2,z3],[a1,a2,a3]]
            
        
           
    
    
    
    def fit(self,X,Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
        
        w1=np.random.rand(3,9)
        w2=np.random.rand(3,3)
        
        w3=np.random.rand(1,3)
        
        weights=[w1,w2,w3]
        
        
        bias1=[np.random.rand() for i in range(3)]
        bias1=np.array(bias1)
        bias2=[np.random.rand() for i in range(3)]
        bias2=np.array(bias2)
        bias3=[np.random.rand() for i in range(1)]
        bias3=np.array(bias3)
        
        bias=np.array([bias1,bias2,bias3])
        for epoch in range(self.epochs):
            for i in range(len(X)):
                list1=self.foward(weights,X[i],bias)
                self.backprop(weights,Y,list1[0],list1[1],i,X[i],bias)
            
            
        
        
        self.copy_weights=weights
        self.copy_bias=bias
    
    
    
            
    
    
    def predict(self,X):
        yhat=[]
        for i in range(len(X)):
          list1=self.foward(self.copy_weights,X[i],self.copy_bias)
          
          bias1=self.copy_bias[0]
          bias2=self.copy_bias[1]
          bias3=self.copy_bias[2]
          
          yhat.append(list1[1][2][0])
        return yhat
    
    def CM(self,y_test,y_test_obs):
        '''
        Prints confusion matrix 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model
        
        '''
        
        for i in range(len(y_test_obs)):
          if(y_test_obs[i]>0.6):
            y_test_obs[i]=1
          else:
            y_test_obs[i]=0
        print(y_test_obs)
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
        
        for i in range(len(y_test)):
          if(y_test[i]==1 and y_test_obs[i]==1):
            tp=tp+1
          if(y_test[i]==0 and y_test_obs[i]==0):
            tn=tn+1
          if(y_test[i]==1 and y_test_obs[i]==0):
            fp=fp+1
          if(y_test[i]==0 and y_test_obs[i]==1):
            fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp
        try:
            p= tp/(tp+fp)
            r=tp/(tp+fn)
            f1=(2*p*r)/(p+r)
        except: print("ERROR")
        
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        print("ACCURACY :",(tp+tn)/(tp+tn+fp+fn))

obj=NN()
#from sklearn import linear_model
x = df.iloc[: , :9].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
skip=[33]
for i in range(75,76):
    if i in skip:
        continue
    
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.4,random_state=3,stratify=y)
    obj.fit(X_train,Y_train)
    
    yhat=obj.predict(X_test)
    
    yhat=np.array(yhat)
    
    
    
    obj.CM(Y_test,yhat)
