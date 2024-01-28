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

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# x = np.array([1.0, 0.5])
# w1 = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
# b1 = np.array([0.1,0.2,0.3])

# print(w1.shape)
# print(x.shape)
# print(b1.shape)

# a1 = np.dot(x,w1) + b1

# Z1 = sigmoid(a1)
# print(Z1)
# print(a1)

# W2 = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
# b2 = np.array([0.1,0.2])

# print(Z1.shape)
# print(W2.shape)
# print(b2.shape)

# a2 = np.dot(Z1,W2) + b2
# Z2 = sigmoid(a2)

# def identity_function(x):
#     return x

# W3 = np.array([[0.1,0.3], [0.2,0.4]])
# b3 = np.array([0.1,0.2])

# a3 = np.dot(Z2,W3) + b3
# Y = identity_function(a3)


# ## 구현 정리
# ## init_network() : 가중치와 편향을 초기화하고 딕셔너리 변수 network에 저장한다
# ## 딕셔너리 변수 network에는 각 층에 필요한 매개변수(가중치, 편향)을 저장한다

# def init_network():
#     network = {}
#     network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
#     network['b1'] = np.array([0.1,0.2,0.3])
#     network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
#     network['b2'] = np.array([0.1,0.2])
#     network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
#     network['b3'] = np.array([0.1,0.2])

#     return network

# ## forward() : 입력 신호를 출력으로 변환하는 처리 과정을 모두 구현한다
# ## 입력 신호가 순방향(입력에서 출력 방향)으로 전달됨에 주의하자

# def forward(network, x):
#     W1, W2, W3 = network['W1'],network['W2'],network['W3']
#     b1, b2, b3 = network['b1'],network['b2'],network['b3']

#     a1 = np.dot(x,W1) + b1
#     Z1 = sigmoid(a1)
#     a2 = np.dot(Z1,W2) + b2
#     Z2 = sigmoid(a2)
#     a3 = np.dot(Z2,W3) + b3
#     Y = identity_function(a3)

#     return Y

# network = init_network()
# x = np.array([1.0,0.5])
# y = forward(network, x)
# print(y)

## 출력층 설계하기
## 일반적으로 회귀에는 항등 함수를, 분류에는 소프트맥스 함수를 사용한다
## 항등 함수 : 입력을 그대로 출력한다
## 소프트맥스 함수 : 입력 신호를 정규화하여 출력한다

## 기계학습 문제는 분류(classification)와 회귀(regression)로 나눌 수 있다
## 분류(classification) : 데이터가 어느 클래스에 속하느냐 문제
## 회귀(regression) : 입력 데이터에서 (연속적인) 수치를 예측하는 문제

## 항등 함수(identity function) : 입력을 그대로 출력하는 함수
## 출력층에서 항등 함수를 사용하면 입력 신호가 그대로 출력 신호가 된다

## 소프트맥스 함수(softmax function) : 입력 값을 정규화하여 출력한다

## yk = exp(ak) / sigma(i=1~n) exp(ai)

## n : 출력층의 뉴런 수
## yk : k번째 출력
## ak : k번째 출력의 입력 신호


## 소프트맥스 함수 구현하기

# a = np.array([0.3,2.9,4.0])

# exp_a = np.exp(a) ## 지수 함수
# print(exp_a)

# sum_exp_a = np.sum(exp_a) ## 지수 함수의 합
# print(sum_exp_a)

# y = exp_a / sum_exp_a
# print(y)

# ## exp_a 함수란? : 밑(base)이 자연상수 e인 지수 함수


# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a

#     return y

# a = np.array([1010,1000,990])
# print(np.exp(a) / np.sum(np.exp(a)))

 ## 소프트맥스 함수의 계산 결과는 모두 0이 되어버린다
## nan : not a number
## 해결책 : 입력 신호 중 최댓값을 빼주면 올바르게 계산할 수 있다

# c = np.max(a)  ## c = 1010
# print(a-c)

# print(np.exp(a-c) / np.sum(np.exp(a-c)))

## 소프트맥스 함수 구현 정리
## 1. 입력 신호 중 최댓값을 빼준다(오버플로 대책)
## 2. exp() 함수를 적용한다
## 3. exp() 함수의 출력을 모두 더한다
## 4. 3의 결과로 나온 값을 분모와 분자로 나눈다

# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a-c) ## 오버플로 대책
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a

#     return y

# ## 소프트맥스 함수의 특징
# ## 소프트맥스 함수의 출력은 0에서 1.0 사이의 실수이다

# a = np.array([0.3,2.9,4.0])
# y = softmax(a)
# print(y)
# print(np.sum(y)) ## 소프트맥스 함수의 출력은 0에서 1.0 사이의 실수이다

## 소프트맥스 함수의 출력 총합은 1이다
##  -> 이 성질 덕분에 소프트맥스 함수의 출력을 '확률'로 해석할 수 있다

## 기계 학습의 문제 풀이 학습과 추론의 두 단계
## 학습 : 모델을 학습하하는 것 -> 훈련 데이터를 사용하여 가중치 매개변수를 학습하는 것
## 추론 : 앞서 학습한 모델로 미지의 데이터에 대해서 추론(분류)하는 것

## 신경망에서는 학습 때는 Softmax 함수를 사용하고, 
## 추론 때는 Softmax 함수를 생략하는 것이 일반적이다

# import sys, os
# sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# from dataset.mnist import load_mnist
# from PIL import Image

# (x_train, t_train), (x_test, t_test) = \
#     load_mnist(flatten=True, normalize=False)

# ## 각 데이터의 형상 출력
# print(x_train.shape) 
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

## load_mnist() 함수는 읽은 MNIST 데이터를 
## (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블) 형식으로 반환한다

## 인수로는 normalize, flatten, one_hot_label 세 가지를 설정 가능 

## normalize : 입력 이미지의 픽셀 값을 0.0 ~ 1.0 사이의 값으로 정규화할지 정한다
## false로 설정시 입력 이미지의 픽셀은 원래 값 그대로 0 ~ 255 사이의 값을 유지한다

## flatten : 입력 이미지를 평탄하게, 즉 1차원 배열로 만들지를 정한다
## false로 설정시 입력 이미지를 1 x 28 x 28의 3차원 배열로,
## true로 설정시 784개의 원소로 이루어진 1차원 배열로 저장한다

## one_hot_label : 원-핫 인코딩 형태로 저장할지 정한다
## 원-핫 인코딩이란 정답을 뜻하는 원소만 1이고 나머지는 0인 배열이다
## one_hot_label이 false면 '7'이나 '2'와 같은 레이블을 숫자 그대로 저장한다

import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
# from PIL import Image
import pickle
from common.functions import sigmoid, softmax

# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img)) ## numpy로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
#     pil_img.show()

# (x_train, t_train), (x_test, t_test) = \
#     load_mnist(flatten=True, normalize=False)

# img = x_train[0]
# label = t_train[0]
# print(label) # 5

# print(img.shape) # (784,)
# img = img.reshape(28,28) # 원래 이미지의 모양으로 변형
# print(img.shape) # (28,28)

# img_show(img)

## MNIST 데이터셋을 이용한 신경망의 추론 처리

## 신경망의 추론 처리 구성 
## 1. 입력층 뉴런 : 784개(이미지 크기 : 28 x 28)
## 2. 출력층 뉴런 : 10개(0~9까지의 숫자를 구분)

## 입력층 뉴런 784개 -> 은닉층 뉴런 50개 -> 은닉층 뉴런 100개 -> 출력층 뉴런 10개

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False, one_hot_label=False)

    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f: ## 가중치와 편향 매개변수를 sample_weight.pkl에 저장
        network = pickle.load(f)

    return network

def predict(network, x): ## 입력 x가 주어졌을 때의 출력 y를 구하는 처리 과정
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    Z1 = sigmoid(a1)
    a2 = np.dot(Z1,W2) + b2
    Z2 = sigmoid(a2)
    a3 = np.dot(Z2,W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)): ## len(x) : 10,000
    y = predict(network, x[i])
    p = np.argmax(y) ## 확률이 가장 높은 원소의 인덱스를 얻는다
    if p == t[i]:
        accuracy_cnt += 1


print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


