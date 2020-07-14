#!/usr/bin/env python
# coding: utf-8

# In[ ]:


abs(x)는 어떤 숫자를 입력받았을 때, 그 숫자의 절댓값을 돌려주는 함수이다.


# In[ ]:


all(x)는 반복 가능한(iterable) 자료형 x를 입력 인수로 받으며 
이 x가 모두 참이면 True, 거짓이 하나라도 있으면 False를 돌려준다.

※ 반복 가능한 자료형이란 for문으로 그 값을 출력할 수 있는 것을 의미한다. 
리스트, 튜플, 문자열, 딕셔너리, 집합 등이 있다.


# In[ ]:


any(x)는 x 중 하나라도 참이 있으면 True를 돌려주고, 
x가 모두 거짓일 때에만 False를 돌려준다. all(x)의 반대이다.


# In[ ]:


chr(i)는 아스키(ASCII) 코드 값을 입력받아 그 코드에 
해당하는 문자를 출력하는 함수이다.


# In[ ]:


dir은 객체가 자체적으로 가지고 있는 변수나 함수를 보여 준다. 
다음 예는 리스트와 딕셔너리 객체 관련 함수(메서드)를 보여 주는 예이다. 


# In[ ]:


divmod(a, b)는 2개의 숫자를 입력으로 받는다. 
그리고 a를 b로 나눈 몫과 나머지를 튜플 형태로 돌려주는 함수이다.


# In[ ]:


enumerate는 "열거하다"라는 뜻이다. 
이 함수는 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 
인덱스 값을 포함하는 enumerate 객체를 돌려준다.


# In[ ]:


eval(expression )은 실행 가능한 문자열(1+2, 'hi' + 'a' 같은 것)을 입력으로 받아 
문자열을 실행한 결괏값을 돌려주는 함수이다.

보통 eval은 입력받은 문자열로 파이썬 함수나 클래스를 동적으로 실행하고 싶을 때 
사용한다.


# In[ ]:


filter 함수는 첫 번째 인수로 함수 이름을, 두 번째 인수로 그 함수에 차례로 들어갈 
반복 가능한 자료형을 받는다. 그리고 두 번째 인수인 반복 가능한 자료형 요소가 
첫 번째 인수인 함수에 입력되었을 때 반환 값이 참인 것만 묶어서(걸러 내서) 돌려준다.


# In[ ]:


input([prompt])은 사용자 입력을 받는 함수이다. 
매개변수로 문자열을 주면 다음 세 번째 예에서 볼 수 있듯이 
그 문자열은 프롬프트가 된다


# In[ ]:


int(x)는 문자열 형태의 숫자나 소수점이 있는 숫자 등을 정수 형태로 돌려주는 함수로, 
정수를 입력으로 받으면 그대로 돌려준다.


# In[ ]:


map(f, iterable)은 함수(f)와 반복 가능한(iterable) 자료형을 입력으로 받는다. 
map은 입력받은 자료형의 각 요소를 함수 f가 수행한 결과를 묶어서 돌려주는 함수이다.


# In[ ]:


pow(x, y)는 x의 y 제곱한 결괏값을 돌려주는 함수이다.


# In[ ]:


round(number[, ndigits]) 함수는 숫자를 입력받아 반올림해 주는 함수이다.


# In[ ]:


zip(*iterable)은 동일한 개수로 이루어진 자료형을 묶어 주는 역할을 하는 함수이다.


# In[ ]:


sort()함수
#리스트를 정렬된 상태로 변경

sorted()함수
#이터러블 객체로부터 정렬된 리스트를 생성

list.sort()와 내장함수 sorted()는 모두 reverse 매개변수를 받는다. 
reverse 변수는 부울형으로 True 이면 내림차순이 된다. 


# In[ ]:


replace()함수
replace(old, new, [count]) -> replace("찾을값", "바꿀값", [바꿀횟수])


# In[ ]:





# In[ ]:





# In[ ]:





# # Numpy

# In[ ]:


import numpy as np
import pandas as pd
import numpy.random as npr


# In[ ]:


#배열생성
np.arange() #1차원 배열생성


# In[ ]:


#배열의 차원 변경
np.arange(d).reshape(a,b,c) 
#d까지의 숫자로 배열을 생성하고 a,b,c 형태로 배열의 차원변경, 인수가 2개면 2차원 3개면 3차원


# In[ ]:


np.random.seed(0) #시드설정


# In[ ]:


np.random.rand(a) #a개만큼 난수생성


# In[ ]:


np.random.shuffle(x) #x의 데이터 섞기


# In[ ]:


np.random.choice(a, size=None, replace=True, p=None)
#a 배열이면 원래의 데이터, 정수이면 arrange(a) 명령으로 데이터생성
#size = 정수, 샘플 숫자
#replace = True : 복원추출 False : 비복원추출
#p = 배열, 각 데이터가 선택될 수 있는 확률
#p의 경우 리스트를 만들어서 선택확률을 다르게 할 수 있다.


# In[ ]:


#난수 생성
#rand: 0부터 1사이의 균일 분포
#randn: 가우시안 표준 정규 분포
#randint: 균일 분포의 정수 난수
np.random.randint(low, high=None, size=None)
#high 입력 없을시 0과 low사이의 숫자를 입력한다. 
#size는 난수의 숫자


# In[ ]:


unique() #명령은 데이터에서 중복된 값을 
#제거하고 중복되지 않는 값의 리스트를 출력
return_counts #인수를 True 로 설정하면 각 값을 가진 데이터 갯수도 출력한다.
np.unique(a, return_counts=True)

#그러나 unique 명령은 데이터에 존재하는 값에 대해서만 갯수를 세므로 
#데이터 값이 나올 수 있음에도 불구하고 데이터가 하나도 없는 경우에는 
#정보를 주지 않는다. 예를 들어 주사위를 10번 던졌는데 6이 한 번도 나오지
#않으면 이 값을 0으로 세어주지 않는다.

#따라서 데이터가 주사위를 던졌을 때 나오는 수처럼 특정 범위안의 수인 경우에는 
#bincount에 minlength 인수를 설정하여 쓰는 것이 더 편리하다. 
#이 때는 0 부터 minlength - 1 까지의 숫자에 대해 각각 카운트를 한다. 
#데이터가 없을 경우에는 카운트 값이 0이 된다.


# In[ ]:


#Arg를 이용한 Sorting
np.argsort(x) #x배열의 작은 값부터 순서대로 데이터의 index를 반환해줌


# In[ ]:


np.concatenate()
np.concatenate((array_1, array_2), axis=1 or 0)
#행과 열 기준으로 배열 합치기


# In[ ]:


x.flatten()
#1차원 배열로 만든다.


# In[ ]:


a.copy() 
#a를 복제한 데이터 


# In[ ]:


.flat = value 
#flat에 인덱스 지정해서 정해진 위치값 바꿀수도 있다.
#value값으로 전체 value값 바꾸기


# In[ ]:


# normal(평균값, 표준편차, 사이즈=n)
data = npr.normal(100, 15, 25)
data


# In[ ]:


np.add.accumulate(data)
#data의 값들의 누적합을 보여준다.


# In[ ]:


전치연산 .T


# # Pandas

# In[ ]:


pd.to_numeric(df['col_name']) #한개의 문자열컬럼을 수치형으로 만들기

df[['col_name', 'col_name2']].apply(pd.to_numeric) #여러개의 문자열칼럼을 수치형으로 만들기


# In[ ]:


#DataFrame 내 모든 문자열 칼럼을 int, float, 'str'로 한꺼번에 변환하기
df.astype(int, float, 'str')  

#DataFrame 내 문자열 칼럼별로 int, float 데이터 형식 개별 지정해서 숫자형으로 변환하기
df.astype({'col_str_1': int, 'col_str_2': np.float}) 


# In[ ]:


#split 함수는 어떠한 기준을 가지고 문자열들을 나눈 다음에 리스트 형식으로 반환합니다. 
.str.split()  #바꾸고 싶은 문자열 . 함수. (인자) 인자가 문자열인 경우에는 ''으로 씌워줘야한다.


#해당 열의 문자열중 첫번째를 가져온다. 주로 날짜데이터에서 연월일 가져올때 혹은 성과 이름을 가져올때 사용된다.
.str.get(0)

#예시
name_split = df["full_name"].str.split(" ")
df["first_name"] = name_split.str.get(0)
df["last_name"] = name_split.str.get(1)


#문자열 결합하는 함수 :  join()
'원하는 기준' . join( 바꾸고싶은 문자열 또는변수 )


# In[ ]:


#get() 함수를 이용해서 에러 없이 value 가져오기
.get() #딕셔너리에서 사용


# In[ ]:


##where 메서드

where 메서드는 특정 조건에 맞는 데이터들을 선택하여 출력하는데 사용될 수 있다.
조건식을 입력받는데, 조건식에 True에 해당하는 데이터들을 출력한다.
적용되는 Series나 DataFrame의 형상과 일치하는 데이터를 출력한다.

where() 메서드 사용형식

Series.where(cond, other=None, inplace=False, axis=None)

DataFrame.where(cond, other=None, inplace=False, axis=None)

cond는 조건식을 입력받는다. 
메서드가 사용되는 배열과 형상이 동일한 불린배열(boolean array)을 입력해도 무방하다.

other는 조건식이 만족되지 않을 경우 채울 값을 입력받는다.
기본값은 None이고 이때 NaN을 출력한다.

inplace가 True일 경우 where메서드 적용 시 출력값으로 원본 배열을 교체한다.


참고사항(대괄호[조건식] 대신 where 메서드간의 다른 점)

  Series에서 적용 시

    ○ 대괄호[조건식]은 조건식이 True가 되는 결과만 출력한다.

    ○ where 메서드는 조건식이 False이 되면 other에 입력된 값을 출력하거나 NaN을 출력하여 기존 배열과 형상을 맞춘다.

    ○ 즉, 기존 배열과 형상을 똑같이 유지할 필요가 있을 경우에는 where를 아니면 대괄호[조건식]을 사용하면 된다.

  DataFrame에서 적용 시

    ○ 조건식의 결과가 False일 때 대괄호[조건식]은 NaN값을 넣고 DataFrame을 출력한다.

    ○ 조건식의 결과가 False일 때 where 메서드는 기본값으로 NaN값 넣고 DataFrame을 출력하지만, 

       other변수에 값을 입력함으로써 사용자가 원하는 값이 넣어진 DataFrame을 출력할 수 있다.

참고사항 : numpy에서의 where 함수와 pandas에서의 where 메서드는 약간 사용법이 다르니 사용에 주의하자.


##mask 메서드 

mask 메서드는 특정 조건에 맞지 않는 데이터들을 선택하여 출력하는데 사용될 수 있다. 

 ○ 이 메서드는 조건식을 입력받는데, 조건식에 False에 해당하는 데이터들을 출력한다.

 ○ 이 메서드는 적용되는 Series나 DataFrame의 형상과 일치하는 데이터를 출력한다.

 ○ where 메서드와는 반대이다.    
    
mask() 메서드 사용형식)

Series.mask(cond, other=None, inplace=False, axis=None)

DataFrame.mask(cond, other=None, inplace=False, axis=None)    
    
cond는 조건식을 입력받는다.
  ○ 메서드가 사용되는 배열과 형상이 동일한 불린배열(boolean array)을 입력해도 무방하다.
other는 조건식이 만족되지 않을 경우 채울 값을 입력받는다.
  ○ 기본값은 None이고 이때 NaN을 출력한다.
inplace가 True일 경우 mask메서드 적용 시 출력값으로 원본 배열을 교체한다.


참고사항(mask와 where 메서드간의 공통점과 차이점)

  차이점: Where은 조건식이 True일 때 데이터를 살려두나, mask는 이와 반대로 False일 때 데이터를 살려둔다.

  공통점: 그 외에 where메서드에서 적용된 입력변수들은 동일하게 적용 가능하다. 

----------------------------------------------------------------------------------------------------------    
DataFrame.where(self, cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False)

cond에 입력된 값이 True일 경우 데이터를 그대로 두고, False일 경우 값을 대체하여 출력한다.


cond: boolean Series/DataFrame, array-like, or callable 

cond가 True일 경우 기존값을 유지시킨다.

cond가 False일 경우, 기존값을 other값으로 대체한다. 

cond가 callable일 경우 Series/DataFrame에서 계산되며, 불린 Series/DataFrame 혹은 배열을 반환한다.

callable은 반드시 입력 Series/DataFrame을 바꿔서는 안된다. (pandas는 이를 체크하지는 않는다.)

other: scalar, Series/DataFrame, or callable

cond가 False일 경우 해당 값을 other에 입력된 값으로 대체한다.

만약 callable이 입력될 경우 Series/DataFrame에서 계산이 되며, 반드시 스칼라나 Series/DataFrame값을 반환하여야 한다.

callable은 반드시 입력 Series/DataFrame을 변경해서는 안된다. (pandas는 이를 체크하지는 않는다.)

inplace: bool

해당 위치에서 연산을 수행한 후 대체할지 여부를 결정짓는다.

기본값은 False이다.

axis: int

메서드가 적용될 축을 선택한다.(각 축에 따라 다른 값을 채워 넣을 경우)

기본값은 None이다.

level : int

메서드가 적용될 레벨을 선택한다. (멀티인덱스 인 경우)

기본값은 None이다.

errors: str, {'raise', 'ignore'}

에러를 발생시킬 것인지, 발생시키지 않고 원본배열을 반환할 것인지 결정해준다.

기본값은 'raise'이다.

  - 'raise': 예외를 발생시키는 것을 허가한다.

  - 'ignore': 예외를 무시한다. 에러 발생시 원본 객체를 반환한다.

try_cast: bool

만약 가능하다면, 입력된 타입으로 결과를 캐스팅 하는 것을 시도한다.

기본값은 False이다.


 DataFrame.mask(self, cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=False)

cond에 입력된 값이 False일 경우 데이터를 그대로 두고, True일 경우 값을 대체하여 출력한다.


cond: boolean Series/DataFrame, array-like, or callable 

cond가 False일 경우 기존값을 유지시킨다.

cond가 True일 경우, 기존값을 other값으로 대체한다. 

cond가 callable일 경우 Series/DataFrame에서 계산되며, 불린 Series/DataFrame 혹은 배열을 반환한다.

callable은 반드시 입력 Series/DataFrame을 바꿔서는 안된다. (pandas는 이를 체크하지는 않는다.)


other: scalar, Series/DataFrame, or callable

cond가 True일 경우 해당 값을 other에 입력된 값으로 대체한다.

만약 callable이 입력될 경우 Series/DataFrame에서 계산이 되며, 반드시 스칼라나 Series/DataFrame값을 반환하여야 한다.

callable은 반드시 입력 Series/DataFrame을 변경해서는 안된다. (pandas는 이를 체크하지는 않는다.)


inplace: bool

해당 위치에서 연산을 수행할 지 여부를 결정짓는다.

기본값은 False이다.


axis: int

메서드가 적용될 축을 선택한다.(각 축에 따라 다른 값을 채워 넣을 경우)

기본값은 None이다.


level : int

메서드가 적용될 레벨을 선택한다. (멀티인덱스인 경우)

기본값은 None이다.


errors: str, {'raise', 'ignore'}
에러를 발생시킬 것인지, 발생시키지 않고 원본배열을 반환할 것인지 결정해준다.

기본값은 'raise'이다.

  - 'raise': 예외를 발생시키는 것을 허가한다.

  - 'ignore': 예외를 무시한다. 에러 발생시 원본 객체를 반환한다.

try_cast: bool

만약 가능하다면, 입력된 타입으로 결과를 캐스팅 하는 것을 시도한다.

기본값은 False이다.    


# In[ ]:



 인덱싱 방법들

아래 표는 pandas에서 처리하는 대표적인 인덱싱 방법을 보여준다. 

별다른 설명이 없다면 Series는 스칼라 값을 인덱싱하고, DataFrame는 행을 인덱싱한다.

  ○ 예를들어, 행을 선택하는 것은 DataFrame의 행에 해당하는 Series를 반환한다.


 대괄호 [ ] 

 레이블이나 위치 정수를 활용하여 인덱싱을 한다.

 레이블 인덱싱 .loc[label]

 레이블을 기반으로 인덱싱을 한다.

 위치 인덱싱 .iloc[position]

 위치 정수를 기반으로 인덱싱을 한다.

 슬라이싱  [ : ]

 슬라이싱을 활용하여 인덱싱을 한다.

 불린 인덱싱 [ bool_vec ] 

 불린 벡터를 활용하여 인덱싱을 한다.

   ○ 불린 연산(<, >, ==, != 등)을 활용하여 조건이 참인 값들을 인덱싱한다.

   ○ Series에 적용시 Series를 반환한다. 

   ○ DataFrame에 적용시 DataFrame을 반환한다. 

 레이블 인덱싱 .at[label]

 레이블을 기반으로 인덱싱을 한다.

   ○ DataFrame과 Series 상관없이 하나의 스칼라값에 접근한다.

 위치 인덱싱 .iat[position]

 위치 정수를 기반으로 인덱싱을 한다.

   ○ DataFrame과 Series 상관없이 하나의 스칼라값에 접근한다.

.ix 인덱서 .ix[ ]

 레이블이나 위치 정수를 활용하여 인덱싱을 한다.
   ○ 레이블과 위치정수 둘 다 사용할 수는 있다.

   ○ 유용할 것 같지만 둘 다 사용할 수 있다는 점 때문에 혼선이 생길 가능성이 높아지므로 사용은 지양된다.
    
    
# 나이가 10세 미만(0~9세) 또는 60세 이상인 승객의 age, sex, alone 열만 선택 #타이타닉 예시
mask3 = (titanic.age < 10) | (titanic.age >= 60)
df_under10_morethan60 = titanic.loc[mask3, ['age', 'sex', 'alone']]
df_under10_morethan60.head()    


# In[ ]:


sort_values()

#df의' 열(by='sequence')을 기준으로 index(axis=0) 오름차순 정렬하기

df.sort_values(by=['sequence'], axis=0)

#열 이름을 (알파벳 순서로) 정렬하기 :  axis=1

df.sort(axis=1)

#인덱스 기준으로 정렬하기

sort_index()


# In[ ]:


map(함수, 리스트)
예제 >>> list(map(lambda x: x ** 2, range(5)))     # 파이썬 2 및 파이썬 3
[0, 1, 4, 9, 16]

filter(함수, 리스트)
예제 >>> list(filter(lambda x: x < 5, range(10))) # 파이썬 2 및 파이썬 3
[0, 1, 2, 3, 4]

df.apply(lambda x: x.max() - x.min())
titanic['survived'] = titanic['survived'].apply(lambda x : 'survived' if x == 1 else 'died')

df['Child_Adult'] = df['Age'].apply(lambda x: 'Child' if x <= 18 else 'Adult')
df['Age_cat'] = df.apply(lambda x: x['Age'] + x['Fare'] if x['Age'] + x['Fare'] <= 18 else (x['Age'] + x['Fare'] if x['Age'] <= 60 else 'Elderly'), axis=1


# In[ ]:


#pandas에서 제공하는 date_range함수는 datetime 자료형으로 구성된, 날짜/시간 함수


# In[ ]:


groupby()


# In[ ]:


df.agg(['mean', 'min', 'max', 'median', 'std'])


# In[ ]:


# isin() 메서드 활용하여 동일한 조건으로 추출
#isin 구문은 열이 list의 값들을 포함하고 있는 모든 행들을 골라낼 때 주로 쓰인다. 

#예시
isin_filter = titanic['sibsp'].isin([3, 4, 5])
df_isin = titanic[isin_filter]
df_isin.head()



#####불린인덱싱

df.loc[df.Age < 5].sort_values(by = 'Age')

df.loc[df.Age > 60].sort_values(by=['Pclass', 'Age'], ascending = True)

# 함께 탑승한 형제 또는 배우자의 수가 3, 4, 5인 승객만 따로 추출 - 불린 인덱싱
mask3 = titanic['sibsp'] == 3
mask4 = titanic['sibsp'] == 4
mask5 = titanic['sibsp'] == 5
df_boolean = titanic[mask3 | mask4 | mask5]
df_boolean.head()

#나이가 10세 미만(0~9세) 또는 60세 이상인 승객의 age, sex, alone 열만 선택
mask3 = (titanic.age < 10) | (titanic.age >= 60)
df_under10_morethan60 = titanic.loc[mask3, ['age', 'sex', 'alone']]
df_under10_morethan60.head()

# 나이가 10세 미만(0~9세)이고 여성인 승객만 따로 선택
mask2 = (titanic.age < 10) & (titanic.sex == 'female')
df_female_under10 = titanic.loc[mask2, :]

df_female_under10.head()

# 나이가 10대(10~19세)인 승객만 따로 선택
mask1 = (titanic.age >= 10) & (titanic.age < 20)
df_teenage = titanic.loc[mask1, :]

df_teenage.head()


# In[ ]:


#결측치의 대부분은 평균값으로 대체한다 데이터 손실을 막기위해


#결측치 채우기
.fillna(0)

# notnull() 메서드로 누락 데이터 찾기
df.head().notnull()

# isnull() 메서드로 누락 데이터 개수 구하기
df.head().isnull().sum()

# isnull() 메서드로 누락 데이터 찾기
df.isnull()

#결측치삭제
df.dropnp()

# NaN 값이 500개 이상인 열을 모두 삭제 - deck 열(891개 중 688개의 NaN 값)
df_thresh = df.dropna(axis=1, thresh=500)  
df_thresh.columns

# age 열에 나이 데이터가 없는 모든 행을 삭제 - age 열(891개 중 177개의 NaN 값)
df_age = df.dropna(subset=['age'], how='any', axis=0)  
len(df_age)

# age 열의 NaN값을 다른 나이 데이터의 평균으로 변경하기
mean_age = df['age'].mean(axis=0)   # age 열의 평균 계산 (NaN 값 제외)
df['age'].fillna(mean_age)
df['age'].head(10)


# In[ ]:


# 데이터프레임 전체 행 데이터 중에서 중복값 찾기
df.duplicated()

# 데이터프레임에서 중복 행을 제거
df.drop_duplicates()


# In[ ]:


# horsepower 열의 누락 데이터('?') 삭제하고 실수형으로 변환
df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
df['horsepower'] = df['horsepower'].astype('float')      # 문자열을 실수형으로 변환

# 3개의 bin에 이름 지정
bin_names = ['저출력', '보통출력', '고출력']

# pd.cut 함수로 각 데이터를 3개의 bin에 할당
df['hp_bin'] = pd.cut(df['horsepower'],     # 데이터 배열
                      bins=3,               # bin_dividers 경계 값 리스트
                      labels=bin_names,     # bin 이름
                      include_lowest=True)  # 첫 경계값 포함 


# In[ ]:


pd.concat([df1, df2])


# In[ ]:


pd.merge(df1, df2, how='outer')


# In[ ]:


df3 = df1.join(df2)


# In[ ]:


df['new_Date'] = pd.to_datetime(df['Date'])   #df에 새로운 열로 추가


# In[ ]:


# Timestamp를 Period로 변환

pr_minute = ts_dates.to_period(freq='T')

pr_second = ts_dates.to_period(freq='S')

pr_hour = ts_dates.to_period(freq='H')

pr_minute = ts_dates.to_period(freq='T')

pr_day = ts_dates.to_period(freq='D')

pr_week = ts_dates.to_period(freq='W')

pr_month = ts_dates.to_period(freq='M')

pr_qtr = ts_dates.to_period(freq='Q')

pr_year = ts_dates.to_period(freq='A')

print(pr_minute)
print(pr_second)
print(pr_hour)
print(pr_day)
print(pr_week)
print(pr_month)
print(pr_qtr)
print(pr_year)


# In[ ]:


period_range


# In[ ]:


sub() 함수


# In[ ]:


re.sub()


# In[ ]:


정규표현식 초급: re.compile

