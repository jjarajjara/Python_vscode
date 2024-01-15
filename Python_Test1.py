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
def step_function(x):
    y = x > 0
    return y.astype(np.int) # astype() : 넘파이 배열의 자료형을 변환한다







