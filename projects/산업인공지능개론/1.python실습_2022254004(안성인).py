# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:34:24 2022
@author(id): 안성인(2022254004)
@exam : 1.파티 준비
        2.등산 시간 계산
        3.수하물 비용 계산
        4.정수들의 합
"""
print('작성자(학번) : 안성인(2022254004)')
print('=================================================')
#예제 : 파티 준비
print('예제 : 파티 준비')
number = int(input("참석자의 수를 입력하시오:"))
chickens = number
beers = number*2
cakes = number*4
print("치킨의 수: ", chickens)
print("맥주의 수: ", beers)
print("케익의 수: ", cakes)
print('=================================================')
#예제 : 등산 시간 계산
print('예제 : 등산 시간 계산')
from math import *
time1 = 10/20
hill = sqrt(3**2+4**2)
time2 = hill/10
time3 = hill/30
time4 = 8/20
total = time1+time2+time3+time4
print(total)
#예제 : 수하물 비용 계산
for cnt in range(0, 2):
    print('=================================================')
    print('예제 : 수하물 비용 계산')
    cnt += 1
    weight = float(input("짐의 무게는 얼마입니까? "))
    if weight > 20:
        print("무거운 짐은 20,000원을 내셔야 합니다. ")
    else:
        print("짐에 대한 수수료는 없습니다. ")
        print("감사합니다. ")
print('=================================================')
# 예제 : 정수들의 합
print('예제 : 정수들의 합')
# 반복을 이용한 정수합 프로그램
sum = 0
limit = int(input("어디까지 계산할까요: "))
for i in range(1, limit+1) :
    sum += i
print("1부터 ", limit, "까지의 정수의 합= ", sum)
print('=================================================')