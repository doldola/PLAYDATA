#!/usr/bin/env python
# coding: utf-8

# # 위스콘신 유방암 분류예측

# In[1]:


import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# **데이터 불러오기**

# In[2]:


cancer = load_breast_cancer()

data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data_df.head(3)


# In[4]:


#컬럼들 뽑기
data_df.columns.tolist() 


# In[28]:


#타겟값 확인
print(cancer.target_names)  


# In[29]:


data_df.shape #타겟컬럼 포함 31개의 컬럼


# In[30]:


data_df.info()#타겟컬럼은 int형식 그외 30개 컬럼은 float형식


# In[31]:


data_df.isnull().sum() #결측치 없음


# In[32]:


print(data_df.describe())


# **컬럼별 분포**

# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


for cnt, col in enumerate(data_df):
    try:
        plt.figure(figsize=(10, 5))
        sns.distplot(data_df[col][cancer['target']==1])
        sns.distplot(data_df[col][cancer['target']==0])
        plt.legend([1,0], loc='best')
        plt.title('histogram of features '+str(col))
        plt.show()

        if cnt >= 9: # 9개 칼럼까지만 출력
            break

    except Exception as e:
        pass


# **위의 그래프들을 보면 분포가 다른 그래프들이 보인다 해당 컬럼이 양성과 악성을 판단하는 기준이 될 수 있다**

# In[38]:


data_df['diagnosis'] = cancer.target
#데이터 프레임에 diagnosis(target) 값을 넣어서 상관관계를 분석하고자 한다.


# In[39]:


data_df.corr().tail()


# **상관관계 표를 보면 mean radius : -0.730029 / mean perimeter : -0.742636 / mean area : -0.708984 / mean concave points : -0.776614  위의 값들과 위의 값들은 평균값들인데 평균이 아닌 값들도 높았다.**

# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name) # 한글 출력 설정 부분

cancer_tmp = data_df.copy()

corrmat = cancer_tmp.corr()
top_corr_features = corrmat.index[abs(corrmat["diagnosis"])>=0.3]

# plot
plt.figure(figsize=(13,10))
g = sns.heatmap(data_df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.tight_layout()
plt.show()


# **위의 시각화 자료를 본 결과 worst concave points 가 - 0.79로 diagnosis와 가장 상관성이 높았다.**

# # Decision Tree 모델학습

# In[95]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 학습과 테스트 데이터 셋으로 분리
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2,  random_state= 156)

# DecisionTreeClassifer 학습. 
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
clf


# In[98]:


y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_test , pred)
accuracy


# **튜닝 없이 그냥 Decision Tree 예측률 0.9385964912280702**

# In[49]:


import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# feature importance 추출 
print("Feature importances:\n{0}".format(np.round(clf.feature_importances_, 3)))

# feature별 importance 매핑
for name, value in zip(cancer.feature_names , clf.feature_importances_):
    if value > 0:
        print('{0} : {1:.3f}'.format(name, value))
    else:
        pass
# feature importance를 column 별로 시각화 하기 
sns.barplot(x=clf.feature_importances_ , y=cancer.feature_names)


# **변수 중요도를 본 결과 worst radius가 0.702 로 제일 높았다. 두 번째로 worst concave points가 0.118로 높았다.**

# **아래의 시각화 자료는 위의 변수중요도와 상관관계지표가 모두 높아 worst radius와 worst concave points 두 변수로 결정트리 scatter 시각화**

# In[50]:


from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.title("3 Class values with 2 Features Sample data creation")

# 2차원 시각화를 위해서 feature는 2개, 결정값 클래스는 3가지 유형의 classification 샘플 데이터 생성. 
X_features = cancer.data
y_labels = cancer.target

# plot 형태로 2개의 feature로 2차원 좌표 시각화, 각 클래스값은 다른 색깔로 표시됨. 
plt.scatter(X_features[:, 0], X_features[:, 7], marker='o', c=y_labels, s=25, cmap='rainbow', edgecolor='k')


# In[86]:


import numpy as np

# Classifier의 Decision Boundary를 시각화 하는 함수
def visualize_boundary(model, X, y):
    fig,ax = plt.subplots()
    
    # 학습 데이타 scatter plot으로 나타내기
    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim_start , xlim_end = ax.get_xlim()
    ylim_start , ylim_end = ax.get_ylim()
    
    # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
    model.fit(X, y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행. 
    xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # contourf() 를 이용하여 class boundary 를 visualization 수행. 
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow', clim=(y.min(), y.max()),
                           zorder=1)


# In[93]:


#위의 함수에 적용하기 위해서 데이터변환
X_features = data_df[['mean radius','mean concave points']].as_matrix()


# **특정한 트리 생성 제약없는 결정트리 시각화**

# In[88]:


from sklearn.tree import DecisionTreeClassifier

# 특정한 트리 생성 제약없는 결정 트리의 Decsion Boundary 시각화.
dt_clf = DecisionTreeClassifier().fit(X_features, y_labels)
visualize_boundary(clf, X_features, y_labels)


# **graphviz로 결정트리 시각화(파라미터 없음)**

# In[51]:


from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함. 
export_graphviz(clf, out_file="tree.dot", class_names=cancer.target_names , feature_names = cancer.feature_names, impurity=True, filled=True)


# In[52]:


import graphviz

# 위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook상에서 시각화 
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# **StratifiedKFold로 교차분석**

# In[62]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
skf_sh = StratifiedKFold(n_splits=10, shuffle=True)
skf_sh.get_n_splits(cancer.data, cancer.target)
print(skf_sh)


# In[63]:


for train_index, test_index in skf.split(cancer.data, cancer.target):
    print("Train set: {}\nTest set:{}".format(train_index, test_index))


# In[64]:


clf = DecisionTreeClassifier()
scores = cross_val_score(clf, cancer.data, cancer.target, cv=skf)
print("K Fold Cross validation score")
print(scores)
print("Average Accuracy")
print(scores.mean())


# **교차분석 시행후 예측율 조금 하락 0.9228891193500994**

# In[66]:


from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[1,2,3,4,5,6,7,8,9,10],'min_samples_split':[2,3,4,5,6,7,8,9,10], 'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]}

grid_dclf = GridSearchCV(clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dclf.fit(X_train, y_train)

print('GridSearchCV 최적 하이퍼 파라미터 :', grid_dclf.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf. best_estimator_

# GridSEarchCV의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 및 평가 수행
dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpredictions)
print('테스트 세트에서의 DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy))


# **GridSearchCV 후 예측율 향상 0.9474**

# In[68]:


#normalize = False: 올바르게 분류된 데이터 건수 출력
#normalize = Ture: 올바르게 분류된 데이터 비율 출력

print(accuracy_score(y_test, dpredictions, normalize=True))


# In[70]:


print(classification_report(y_test, dpredictions))


# In[71]:


print(roc_auc_score(y_test, dpredictions))


# In[92]:


from sklearn.tree import DecisionTreeClassifier

# 특정한 트리 생성 제약없는 결정 트리의 Decsion Boundary 시각화.
dt_clf = DecisionTreeClassifier().fit(X_features, y_labels)
visualize_boundary(clf, X_features, y_labels)


# In[91]:


#제약있음
dt_clf = DecisionTreeClassifier(max_depth =  4, min_samples_leaf =  1, min_samples_split = 2).fit(X_features, y_labels)
visualize_boundary(dt_clf, X_features, y_labels)


# In[89]:


from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함. 
export_graphviz(best_dclf, out_file="tree.dot", class_names=cancer.target_names , feature_names = cancer.feature_names, impurity=True, filled=True)


# In[90]:


import graphviz

# 위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook상에서 시각화 
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# In[ ]:





# # 앙상블

# In[3]:


# 개별 모델은 로지스틱 회귀와 KNN 임. 
lr_clf = LogisticRegression()
knn_clf = KNeighborsClassifier(n_neighbors=8)

# 개별 모델을 소프트 보팅 기반의 앙상블 모델로 구현한 분류기 
vo_clf = VotingClassifier( estimators=[('LR', lr_clf),('KNN',knn_clf)] , voting='soft' )

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                    test_size=0.2 , random_state= 156)

# VotingClassifier 학습/예측/평가. 
vo_clf.fit(X_train , y_train)
pred = vo_clf.predict(X_test)
print('Voting 분류기 정확도: {0:.4f}'.format(accuracy_score(y_test , pred)))

# 개별 모델의 학습/예측/평가.
classifiers = [lr_clf, knn_clf]
for classifier in classifiers:
    classifier.fit(X_train , y_train)
    pred = classifier.predict(X_test)
    class_name= classifier.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_test , pred)))


# ## Random Forest

# ### 수정 버전 01: 날짜 2019.10.27일
# 
# **원본 데이터에 중복된 Feature 명으로 인하여 신규 버전의 Pandas에서 Duplicate name 에러를 발생.**  
# **중복 feature명에 대해서 원본 feature 명에 '_1(또는2)'를 추가로 부여하는 함수인 get_new_feature_name_df() 생성**

# **학습/테스트 데이터로 분리하고 랜덤 포레스트로 학습/예측/평가**

# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 결정 트리에서 사용한 get_human_dataset( )을 이용해 학습/테스트용 DataFrame 반환
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state= 156)

# 랜덤 포레스트 학습 및 별도의 테스트 셋으로 예측 성능 평가
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train , y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test , pred)
print('랜덤 포레스트 정확도: {0:.4f}'.format(accuracy))


# **GridSearchCV 로 교차검증 및 하이퍼 파라미터 튜닝**

# In[10]:


from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[100],
    'max_depth' : [6, 8, 10, 12], 
    'min_samples_leaf' : [8, 12, 18 ],
    'min_samples_split' : [8, 16, 20]
}
# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_clf , param_grid=params , cv=2, n_jobs=-1 )
grid_cv.fit(X_train , y_train)

print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))


# **튜닝된 하이퍼 파라미터로 재 학습 및 예측/평가**

# In[11]:


rf_clf1 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=8,                                  min_samples_split=8, random_state=0)
rf_clf1.fit(X_train , y_train)
pred = rf_clf1.predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test , pred)))


# In[17]:


ftr_importances_values


# **개별 feature들의 중요도 시각화**

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

ftr_importances_values = rf_clf1.feature_importances_
ftr_importances = pd.Series(ftr_importances_values,index=data_df.columns  )
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20 , y = ftr_top20.index)
plt.show()


# ##  GBM(Gradient Boosting Machine)

# In[21]:


from sklearn.ensemble import GradientBoostingClassifier
import time
import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state= 156)

# GBM 수행 시간 측정을 위함. 시작 시간 설정.
start_time = time.time()

gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train , y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

print('GBM 정확도: {0:.4f}'.format(gb_accuracy))
print("GBM 수행 시간: {0:.1f} 초 ".format(time.time() - start_time))


# In[22]:


from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[100, 500],
    'learning_rate' : [ 0.05, 0.1]
}
grid_cv = GridSearchCV(gb_clf , param_grid=params , cv=2 ,verbose=1)
grid_cv.fit(X_train , y_train)
print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))


# In[23]:


scores_df = pd.DataFrame(grid_cv.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score',
'split0_test_score', 'split1_test_score']]


# In[24]:


# GridSearchCV를 이용하여 최적으로 학습된 estimator로 predict 수행. 
gb_pred = grid_cv.best_estimator_.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print('GBM 정확도: {0:.4f}'.format(gb_accuracy))


# In[ ]:




