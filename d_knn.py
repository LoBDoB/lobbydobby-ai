import numpy as np
from copy import deepcopy
from dataPreprocessing import DataPreprocessing




class lectureKNN(DataPreprocessing):


    def cosineSimillarity(self, v1, v2):
        A = np.sqrt(np.sum(np.square(v1)))
        B = np.sqrt(np.sum(np.square(v2)))

        return np.dot(v1,v2) / (A*B)


    def recommenderLecture(self, target_uid, lecture):
        recommender = {'prio': [], 'mid': [], 'mino': [], 'else': []}
        cnt, result = 0, []
        target_lecture = self.matrix[target_uid][:12]
        for lect in lecture:
            for idx, lec in enumerate(lect):
                if target_lecture[idx]==0 and lec==10:
                    recommender['prio'].append(idx)
                elif target_lecture[idx]==0 and (lec==5 or lec==4 or lec==3):
                    recommender['mid'].append(idx)
                elif lec>=2 and target_lecture[idx]<lec:
                    recommender['mino'].append(idx)
                elif target_lecture[idx] < lec:
                    recommender['else'].append(idx)
        del target_lecture, lecture

        for key in recommender:
            try:
                recommend = recommender[key]
                tmp = []
                for i in set(deepcopy(recommend)):
                    tmp.append([i, recommend.count(i)])

                tmp.sort(key=lambda x: x[1], reverse=True)

                for rcm in tmp:
                    result.append(self.lectures[rcm[0]])
                    cnt += 1
                    if cnt == 3:
                        return result
            except:
                continue

        return result


    def preprocessingSimilarity(self, uid, lst):
        cnt, lecture_lst = 0, []
        for l in lst:
            if l[1] >= 0.5:
                cnt += 1
                user_uid = l[0]
                lecture_lst.append(self.matrix[user_uid][:12])
                if cnt == 2:
                    return self.recommenderLecture(uid, lecture_lst)
            else:
                return self.recommenderLecture(uid, lecture_lst)


    def newUser(self, new_user):
        new_temp = []
        for k, v in zip(new_user.keys(), new_user.values()):
            new_uid, new_vec = k, v
        for idx, vec in zip(self.matrix.keys(), self.matrix.values()):
            similarity = self.cosineSimillarity(vec, new_vec)
            new_temp.append((idx, similarity))
        new_temp.sort(key=lambda x: x[1], reverse=True)
        self.matrix[new_uid] = new_vec
        print(new_temp[:5])
        return self.preprocessingSimilarity(new_uid, new_temp)


    def existUser(self, user_uid):
        exist_temp = []
        for idx, vec in zip(self.matrix.keys(), self.matrix.values()):
            if idx != user_uid:
                similarity = self.cosineSimillarity(vec, self.matrix[user_uid])
                exist_temp.append((idx, similarity))
        exist_temp.sort(key=lambda x: x[1], reverse=True)
        print(exist_temp[:5])
        return self.preprocessingSimilarity(user_uid, exist_temp)