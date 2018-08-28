# coding: utf-8

import numpy as np
from datetime import datetime

class RNN():
    def __init__(self, word_dim, hidden_dim = 128, bptt_truncate=5):
        # 변수 초기화
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        # weight 초기화
        self.U = np.random.uniform(-np.sqrt(1./self.word_dim), np.sqrt(1./self.word_dim), (self.hidden_dim, self.word_dim)) # input weight
        self.V = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.word_dim, self.hidden_dim)) # output weight
        self.W = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim)) # hideen state weight
        self.bh = np.zeros(self.hidden_dim)
        self.by = np.zeros(self.word_dim)

    def softmax(self, x):
        # softmax function
        a = np.exp(x - np.max(x))
        return a / np.sum(a)
    
    def forward_propagation(self, x):
        # time step 전체 길이
        T = len(x)
        # forward propagation 진행중 모든 state를 저장한다. S_t = U .dot x_t + W .dot s_{t-1}
        # 초기 state는 0 으로 시작함 (s[-1]은 0으로 초기화 되어있음.).
        # 각 time step은 s의 각 row에 저장되며 s[t]는 rnn internal loop time.
        s = np.zeros((T, self.hidden_dim))
        # 각 time step의 output 또한 저장한다.
        o = np.zeros((T, self.word_dim))
        
        for t in np.arange(T):
            # x[t]로 U를 찾음. U와 one-hot vector를 곱한것과 동일함.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]) + self.bh)
            o[t] = self.softmax(self.V.dot(s[t]) + self.by)
        return [o,s]
    
    def total_loss(self, x, y):
        L = 0
        # 모든 데이터에 대해
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # 맞은 단어를 제외한 단어는 cross entropy에서 (1-y)에 의해 0이 되므로 맞은 단어만 계산한다.
            # o vector의 각 단어에 대한 정답 레이블인 y[i] 위치에 있는 output softmax prob를 찾는다.
            correct = o[np.arange(len(y[i])), y[i]]
            # 하나의 cross entropy loss 를 더한다 
            L += -1*np.sum(np.log(correct))
        return L
    
    def loss(self, x, y):
        # 전체 트레이닝 데이터 셋으로 나눈다.
        N = np.sum((len(y_i) for y_i in y))
        return self.total_loss(x,y) / N
    
    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)
        # gradient를 담아둘 변수
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        dLdbh = np.zeros(self.bh.shape)
        dLdby = np.zeros(self.by.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1 # y_hat - y ... y 
        # output back-prop
        for t in np.arange(T):
            dLdV += np.outer(delta_o[t], s[t].T) # shape = word_dim X hidden_dim
            dLdby += delta_o[t]
            # Error에서 나온 초기 delta
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t]**2))
            # bptt // 초기 설정한 bptt_truncate 까지
            # t 시간이 주어지면 t-1, t-2, ... 계산
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1]) # delta_t를 알고있으면 outer를 이용하여 쉽게 계산할수 있다고...
                dLdU[:, x[bptt_step]] += delta_t # 더해주기만해도됨.
                dLdbh += delta_t
                # 다음 step을 위한 delta 업데이트
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1]**2)                
        return [dLdU, dLdV, dLdW, dLdbh, dLdby]

    def sgd_step(self, x, y, learning_rate=0.01):
        dLdU, dLdV, dLdW, dLdbh, dLdby = self.bptt(x,y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        self.bh -= learning_rate * dLdbh
        self.by -= learning_rate * dLdby

    def train(self, x_train, y_train, learning_rate = 0.01, epoch = 100):
        # loss 저장
        losses = []
        num_examples_seen = 0
        for i in range(epoch):
            # loss 평가
            if (i % 1 == 0):
                loss = self.loss(x_train, y_train)
                losses.append((num_examples_seen,loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: loss after num_examples_seen=%d epoch=%d: %f" %(time, num_examples_seen, i, loss))
            # learning rate 조절
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("setting learning rate to %f" %(learning_rate))
        # 트레이닝
            for i in range(len(y_train)):
                # one sgd step
                self.sgd_step(x_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
            
    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis = 1)
