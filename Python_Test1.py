import numpy as np
import matplotlib.pylab as plt

## NAND , OR 게이트 구현
# def AND(x1, x2):
#     x = np.array([x1,x2]) # 입력
#     w = np.array([0.5, 0.5]) # 가중치
#     b = -0.7 # 편향
#     tmp = np.sum(w*x) + b # 가중치와 입력의 곱의 합에 편향을 더한 값
#     if tmp <= 0:
#         return 0
#     else:
#         return 1


# def NAND(x1, x2):
#     x = np.array([x1,x2])
#     w = np.array([-0.5, -0.5])
#     b = 0.7
#     tmp = np.sum(w*x) + b
#     if tmp <= 0:
#         return 0
#     else:
#         return 1

# def OR(x1, x2):
#     x = np.array([x1,x2])
#     w = np.array([0.5, 0.5])
#     b = -0.2
#     tmp = np.sum(w*x) + b 
#     if tmp <= 0:
#         return 0
#     else:
#         return 1


# ## XOR 게이트는 단층 퍼셉트론으로 표현할 수 없다 

# ## 다층 퍼셉트론으로는 표현 가능하다 AND, NAND, OR 게이트를 조합하여 구현해보자 

# def XOR(x1, x2):
#     s1 = NAND(x1,x2)
#     s2 = OR(x1,x2)
#     y = AND(s1,s2)
#     return y  

## 퍼셉트론에서 신경망으로 

## 신경망 -> 입력층 | 은닉층 | 출력층 으로 구성된다

## 편향 : 뉴런이 얼마나 쉽게 활성화 되는지 제어한다
## 가중치 : 각 신호의 영향력을 제어한다     
## 활성화 함수 : 입력 신호의 총합을 출력 신호로 변환하는 함수   
## 계단함수(Step Function) : 임계값을 경계로 출력이 바뀌는 함수

## 시그모이드 함수(sigmaoid function) : 신경망에서 자주 이용하는 활성화 함수 중 하나
## h(x) = 1 / (1 + exp(-x))
## exp(-x)는 e^-x를 뜻함

## 계단함수 구현

# def step_function(x):
#     if x > 0:
#         return 1  
#     else:
#         return 0
    
## 넘파이 배열도 지원하도록 수정
# def step_function(x):
#     y = x > 0
#     return y.astype(np.int) # astype() : 넘파이 배열의 자료형을 변환한다

# x = np.array([-1.0, 1.0, 2.0])
# # print(x)

# y = x > 0
# print(y)

## 넘파이 배열의 자료형을 변환할 때는 astype() 메서드를 이용한다
## 원하는 자료형을 변환할 때 astype(np.int)처럼 np.int를 인수로 지정한다
## (np.변환하고 싶은 인수)

# y = y.astype(np.int)
# print(y)

## 계단 함수의 그래프를 구현해보자 -5.0에서 5.0까지 0.1 간격의 넘파이 배열을 생성
## 책에선 def step_function(x): return np.array(x > 0, dtype=np.int) 로 구현했지만,
## dtype=np.int를 생략해도 잘 작동한다 
## 이유가 뭘까? -> dtype=np.int를 생략하면 np.array()가 자동으로 dtype을 int64로 지정하기 때문이다
## int64는 64비트 정수형을 뜻한다
## 코파일럿 고마워,, 감동이야,,,구글링이 필요가 없네,,,

# def step_function(x):
#     return np.array(x > 0)

# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1) # y축의 범위 지정
# plt.show()


## 시그모이드 함수 구현하기 
## 시그모이드 함수란 'S자 모양'이라는 뜻을 가진다

## 넘파이 배열이어도 올바른 결과가 나온다
## 시그모이드 함수 그래프 그리기
# def sigmaoid(x):
#     return 1 / (1 + np.exp(-x))    
 
# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmaoid(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1) ## y축의 범위 지정
# plt.show()
    # x = np.array([-1.0, 1.0, 2.0])
    # sigmaoid(x)

    # t = np.array([1.0, 2.0, 3.0])
    # 1.0 + t 
    # 1.0 / t


## 계단 함수와 시그모이드 함수 공통점 
## = 비선형 함수 

## sigmoid 함수와 계단 함수 동시에 그리기 성공 !!
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def step_function(x):
#     return np.array(x > 0)

# x = np.arange(-5.0, 5.0, 0.1)
# y1 = step_function(x)
# y2 = sigmoid(x)
# plt.plot(x,y1)
# plt.plot(x,y2, 'k--') ## k-- : 검은색 점선
# plt.ylim(-0.1, 1.1)
# plt.show()

## ReLU 함수 
## 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수

# def relu(x):
#     return np.maximum(0,x) ## maximum() : 두 입력 중 큰 값을 선택해 반환하는 함수

## 책에선 주로 ReLU 함수를 사용한다고 한다

## 다차원 배열

# A = np.array([1,2,3,4])
# print(A)
# print(np.ndim(A)) ## np.ndim() 함수 : 배열의 차원 수를 반환하는 함수
# print(A.shape) ## 배열의 형상을 반환하는 함수
# print(A.shape[0])

## shape() 함수는 튜플을 반환한다
## 배열의 형상이란 각 차원의 요소 수를 튜플로 표시한 것
## 튜플은 (1,2)처럼 괄호로 둘러싸인 '쉼표로 구분된 값'의 나열이다
### 다차원 배열인 경우 ) 배열의 갯수, 배열 안의 원소 갯수 
### 1차원 배열인 경우 ) 배열 안의 원소 갯수, (공백)


# B = np.array([[1,2], [3,4], [5,6]])
# print(B)
# print(np.ndim(B))
# print(B.shape)


## 지긋지긋한 행렬의 곱 
# A = np.array([[1,2], [3,4]])
# print(A.shape)
# B = np.array([[5,6], [7,8]])
# print(B.shape)
# print(np.dot(A,B)) ## np.dot() 함수 : 행렬의 곱을 계산하는 함수


# x = np.array([1,2])
# print(x.shape)
# w = np.array([[1,3,5], [2,4,6]])
# print(w)
# print(w.shape)
# y = np.dot(x,w)
# print(y)

# A = np.array([[1,2,3], [4,5,6]])
# print(A.shape)
# B = np.array([[1,2], [3,4], [5,6]])
# print(B.shape)
# print(np.dot(A,B))

# C = np.array([[1,2], [3,4]])
# print(C.shape)
# print(A.shape)
# print(np.dot(A,C)) ## 행렬의 곱에서는 형상에 주의해야 한다

# A = np.array([[1,2], [3,4], [5,6]])
# print(A.shape)
# B = np.array([7,8])
# print(B.shape)
# print(np.dot(A,B)) ## 1차원 배열을 2차원 배열로 변환하여 계산한다

## 신경망에서의 행렬 곱

# x = np.array([1,2])
# print(x.shape)
# w = np.array([[1,3,5], [2,4,6]])
# print(w)
# print(w.shape)
# y = np.dot(x,w)
# print(y)

## 3층 신경망 구현하기

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([1.0, 0.5])
w1 = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
b1 = np.array([0.1,0.2,0.3])

print(w1.shape)
print(x.shape)
print(b1.shape)

a1 = np.dot(x,w1) + b1

Z1 = sigmoid(a1)
print(Z1)
print(a1)

W2 = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
b2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(b2.shape)

a2 = np.dot(Z1,W2) + b2
Z2 = sigmoid(a2)

def identity_function(x):
    return x

W3 = np.array([[0.1,0.3], [0.2,0.4]])
b3 = np.array([0.1,0.2])

a3 = np.dot(Z2,W3) + b3
Y = identity_function(a3)


## 구현 정리

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    Z1 = sigmoid(a1)
    a2 = np.dot(Z1,W2) + b2
    Z2 = sigmoid(a2)
    a3 = np.dot(Z2,W3) + b3
    Y = identity_function(a3)

    return Y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)
