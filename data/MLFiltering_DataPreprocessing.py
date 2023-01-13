#!/usr/bin/env python
# coding: utf-8

# # Main

# In[10]:


import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[11]:


data = pd.read_csv('preprocess.csv')

data


# In[12]:


def category_str(data, classification=['인문', '사회', '교육', '공학', '자연', '의약', '예체능']):
    copy_data = deepcopy(data)
    for i in range(7):
        copy_data.loc[ copy_data['category']==i , 'category' ] = classification[i]
    return copy_data


# In[13]:


def preprocess_name(text):
    result = ''
    
    # 빈칸제거
    result = text.strip()
    
    # 한글만 추출
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')  # 한글과 띄어쓰기를 제외한 모든 글자
    result = hangul.sub('', result)
    
    # 1차
    remove_words = ['학과', '전공', '학부', '대학', '계열', '단과대학없음', '프로그램', '과']
    result = result.strip().replace(' ', '').strip()
    
    if result in remove_words:
        return '0'
    else:
        return result


# In[5]:


# data1 = category_str(data)

# data1


# In[14]:


data2 = category_str(data, ['인문/사회', '인문/사회', '교육', '공학', '자연', '의약', '예체능'])

data2


# In[15]:


x = data2['name'].to_list()

x


# In[16]:


model = SentenceTransformer('jhgan/ko-sroberta-multitask')
embeddings = model.encode(x)

embeddings.shape


# In[17]:


scaler = StandardScaler()
scale_embeddings = scaler.fit_transform(embeddings)

scale_embeddings.shape


# In[18]:


def pca_decide(n, data=scale_embeddings):
    pca = PCA(n_components=n)  # 주성분을 몇개로 할지 결정
    test = pca.fit_transform(data)
    df = pd.DataFrame(data=test, columns=[f"pca{num+1}" for num in range(n)])
    
    var = pca.explained_variance_
    ratio = pca.explained_variance_ratio_

    results = pd.DataFrame({'설명가능한 분산 비율(고윳값)':var, '기여율':ratio},
                         index = np.array([f"pca{num+1}" for num in range(n)]))
    results['누적기여율'] = results['기여율'].cumsum()
    
    if results['누적기여율'][-1] >= 0.8:
        return results
    

def find_dimension():
    for n in range(3, 100):
        result = pca_decide(n)
        try:
            if result['설명가능한 분산 비율(고윳값)'][-1] >= 0.7:
                print(result)
                return n
        except:
            pass


# In[19]:


dimension = find_dimension()

print(f"{dimension} 차원")


# In[20]:


pca = PCA(n_components=dimension) # 주성분을 몇개로 할지 결정
test = pca.fit_transform(scale_embeddings)
X = pd.DataFrame(data=test, columns=[f"pca{num+1}" for num in range(dimension)])

X


# In[21]:


y = data2['category']

y


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


rfm = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=3)
rfm.fit(x_train.values, y_train.values)


# In[24]:


pred = rfm.predict(x_test.values)

print(y_test.values, '\n')
print(pred)


# In[25]:


acc = accuracy_score(y_test, pred)

print('Acc: {:.2f}%'.format(acc*100))


# # find fail

# In[26]:


y_test


# In[27]:


pred


# In[28]:


y_test.index


# In[29]:


index_list, leng = [], len(y_test)
for yt, pr, i in zip(y_test, pred, [l for l in range(leng)]):
    if yt != pr:
        index_list.append(i)


# In[30]:


# 틀린 과목 리스트
x_name = []
for i in index_list:
    x_name.append(x[i])
    
x_name


# # Seogang Test

# In[31]:


seogang_class = [
    '인문/사회', '인문/사회', '인문/사회', '인문/사회', '인문/사회', '인문/사회', '인문/사회', '인문/사회', '인문/사회', '인문/사회', 
    '인문/사회', '인문/사회', '인문/사회', '인문/사회', '인문/사회', 
    '예체능', '예체능',  '예체능', '예체능', '예체능', '예체능', 
    '자연', '자연', '자연', '자연', 
    '공학', '공학', '공학', '공학', '공학', '공학'
]

seogang_major = [
    '국어국문학', '사학', '철학', '종교학', '영문학부', '유럽문화', '독일문화', '프랑스문화', '중국문화', '일본문화', 
    '사회학', '정치외교학', '심리학', '경제학', '경영학', 
    '국제한국학', '아트&테크놀로지', '신문방송학', '미디어&엔터테인먼트', '글로벌 한국학', '커뮤니케이션학', 
    '수학', '물리학', '화학', '생명과학', 
    '전자공학', '화공생명공학', '컴퓨터공학', '기계공학', '인공지능학', '시스템반도체공학'
]


# In[32]:


majors = []
for word in seogang_major:
    majors.append(preprocess_name(word))
    
majors


# In[33]:


emb = model.encode(majors)

scale_emb = scaler.transform(emb)

scale_emb.shape


# In[34]:


tes = pca.transform(scale_emb)

tes.shape


# In[35]:


pre = rfm.predict(tes)

print(seogang_class, '\n')
print(pre)


# In[36]:


Acc = accuracy_score(seogang_class, pre)

print('Acc: {:.2f}%'.format(Acc*100))

