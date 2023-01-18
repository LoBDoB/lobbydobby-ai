import numpy as np
import pandas as pd
from copy import deepcopy


# # usern => UID 로 들어갈 예정 => UID 형식: ha1515151 => 학교별 약칭 + 학번
# matrix = {'user1':[10, 5, 10, 10, 0, 2, 10, 0, 0, 0, 0, 0, 15, 0, 5, 0, 0, 0],
#           'user2':[10, 0, 10, 10, 10, 10, 0, 0, 10, 5, 10, 0, 5, 10, 5, 0, 0, 0],
#           'user3':[3, 2, 10, 4, 10, 5, 3, 0, 10, 10, 0, 0, 5, 10, 0, 0, 5, 0],
#           'user4':[5, 4, 0, 10, 0, 0, 10, 10, 0, 0, 3, 0, 15, 0, 0, 5, 0, 0],
#           'user5':[0, 5, 10, 10, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 15, 5, 0, 0]}

data = pd.read_csv('knn_data.csv')

matrix = {}
for i in range(100):
    matrix[f'user{i+1}'] = data.iloc[i].to_list()

lectures = {0: '아픈 영혼을 철학으로 치유하기',
            1: '리더의 전략적 의사결정',
            2: '예술계열 캡스톤 디자인 설계',
            3: '액티브시니어를 위한 힐링 요가',
            4: '데이터엔지니어링',
            5: '인공지능과 헬스케어(AI+Health Care)',
            6: '삶과 교육',
            7: '한국어교육학개론',
            8: '자연모사기술',
            9: '생물학적 인간 Ⅱ',
            10: '생활 속의 물리치료',
            11: '생물정보학과 신약개발'}


def cosineSimillarity(v1, v2):

    A = np.sqrt(np.sum(np.square(v1)))
    B = np.sqrt(np.sum(np.square(v2)))

    return np.dot(v1,v2) / (A*B)


def recommenderLecture(target_uid, lecture, recommender={'prio':[], 'mid':[], 'mino':[], 'else':[]}):

    target_lecture = matrix[target_uid][:12]
    for lect in lecture:
        for idx, lec in enumerate(lect):
            if target_lecture[idx]==0 and lec==10:
                recommender['prio'].append(idx)
            elif target_lecture[idx]==0 and lec==5:
                recommender['mid'].append(idx)
            elif target_lecture[idx]==0 and lec==4:
                recommender['mid'].append(idx)
            elif target_lecture[idx]==0 and lec==3:
                recommender['mid'].append(idx)
            elif lec>=3 and target_lecture[idx]<lec:
                recommender['mino'].append(idx)
            elif target_lecture[idx] < lec:
                recommender['else'].append(idx)
    del target_lecture, lecture

    cnt, result = 0, []
    for key in recommender:
        try:
            recommend = recommender[key]
            tmp = []
            for i in set(deepcopy(recommend)):
                tmp.append([i, recommend.count(i)])

            tmp.sort(key=lambda x: x[1], reverse=True)

            for rcm in tmp:
                result.append(lectures[rcm[0]])
                cnt += 1
                if cnt == 3:
                    return result
        except:
            continue

    return result


def preprocessingSimilarity(uid, lst, lecture_lst=[]):
    cnt = 0
    for l in lst:
        if l[1] >= 0.5:
            cnt += 1
            user_uid = l[0]
            lecture_lst.append(matrix[user_uid][6:])
            if cnt == 2:
                return recommenderLecture(uid, lecture_lst)
        else:
            return recommenderLecture(uid, lecture_lst)


def newUser(new_uid, new_vec, new_temp=[]):

    for idx, vec in zip(matrix.keys(), matrix.values()):
        similarity = cosineSimillarity(vec, new_vec)
        new_temp.append((idx, similarity))
    new_temp.sort(key=lambda x: x[1], reverse=True)
    matrix[new_uid] = new_vec

    print(new_temp[:10])
    return preprocessingSimilarity(new_uid, new_temp)


def existUser(user_uid, exist_temp=[]):

    for idx, vec in zip(matrix.keys(), matrix.values()):
        if idx != user_uid:
            similarity = cosineSimillarity(vec, matrix[user_uid])
            exist_temp.append((idx, similarity))
    exist_temp.sort(key=lambda x: x[1], reverse=True)

    print(exist_temp[:10])
    return preprocessingSimilarity(user_uid, exist_temp)


new = {'user101':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 5, 0, 0, 0]}


print('---------- existed User, recommender Top-3 lectures ----------')
print(existUser('user1'), '\n')
print('------------ new User, recommender Top-3 lectures ------------')
print(newUser('user101', new['user101']), '\n')
print('------------------- confirm updated matrix -------------------')
print(matrix)