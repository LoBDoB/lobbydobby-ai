import numpy as np
from copy import deepcopy
from dataPreprocessing import DataPreprocessing




class lectureKNN(DataPreprocessing):


    def cosineSimillarity(self, v1, v2):
        A = np.sqrt(np.sum(np.square(v1)))
        B = np.sqrt(np.sum(np.square(v2)))

        return np.dot(v1,v2) / (A*B)


    def recommenderLecture(self, target_uid, lecture):
        lst = [0 for _ in range(12)]
        prio, mid, mino, els = lst, lst, lst, lst
        cnt, result = 0, []
        target_lecture = self.matrix[target_uid][:12]
        for lect in lecture:
            for idx, lec in enumerate(lect):
                if target_lecture[idx]==0 and lec==10:
                    prio[idx] += 1
                elif target_lecture[idx]==0 and (lec==5 or lec==4 or lec==3):
                    mid[idx] += 1
                elif lec>=2 and target_lecture[idx]<lec:
                    mino[idx] += 1
                elif target_lecture[idx] < lec:
                    els[idx] += 1
        del target_lecture, lecture

        for idx in range(12):  # prio, mid, mino, els
            try:
                prio_tmp, mid_tmp, mino_tmp, els_tmp = [], [], [], []
                if prio[idx] != 0:
                    prio_tmp.append((idx, prio[idx]))
                elif mid[idx] != 0:
                    mid_tmp.append((idx, mid[idx]))
                elif mino[idx] != 0:
                    mino_tmp.append((idx, mino[idx]))
                elif els[idx] != 0:
                    els_tmp.append((idx, els[idx]))

                prio_tmp.sort(key=lambda x: x[1], reverse=True)
                mid_tmp.sort(key=lambda x: x[1], reverse=True)
                mino_tmp.sort(key=lambda x: x[1], reverse=True)
                els_tmp.sort(key=lambda x: x[1], reverse=True)
                tmp = prio_tmp + mid_tmp + mino_tmp + els_tmp

                for i, v in tmp:
                    if i not in result:
                        result.append(self.lectures[i])
                        cnt += 1
                    if cnt == 3:
                        return result
            except:
                continue

        return result


    def preprocessingSimilarity(self, uid, lst):
        cnt, lecture_lst = 0, []
        for l in lst:
            if l[1] >= 0.6:
                cnt += 1
                user_uid = l[0]
                lecture_lst.append(self.matrix[user_uid][:12])
                if cnt == 3:
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
        print('\n', new_temp[:3])
        return self.preprocessingSimilarity(new_uid, new_temp)


    def existUser(self, user_uid):
        exist_temp = []
        for idx, vec in zip(self.matrix.keys(), self.matrix.values()):
            if idx != user_uid:
                similarity = self.cosineSimillarity(vec, self.matrix[user_uid])
                exist_temp.append((idx, similarity))
        exist_temp.sort(key=lambda x: x[1], reverse=True)
        print(exist_temp[:3])
        tmp1, tmp2, tmp3 = [], [], []
        for i, t1, t2, t3 in zip([0,1,2,3,4,5], self.matrix[exist_temp[0][0]][-6:], self.matrix[exist_temp[1][0]][-6:], self.matrix[exist_temp[2][0]][-6:]):
            if t1 != 0:
                if i == 0:
                    tmp1.append('공학')
                elif i == 1:
                    tmp1.append('인문/사회')
                elif i == 2:
                    tmp1.append('예체능')
                elif i == 3:
                    tmp1.append('교육')
                elif i == 4:
                    tmp1.append('자연')
                elif i == 5:
                    tmp1.append('의약')
            if t2 != 0:
                if i == 0:
                    tmp2.append('공학')
                elif i == 1:
                    tmp2.append('인문/사회')
                elif i == 2:
                    tmp2.append('예체능')
                elif i == 3:
                    tmp2.append('교육')
                elif i == 4:
                    tmp2.append('자연')
                elif i == 5:
                    tmp2.append('의약')
            if t3 != 0:
                if i == 0:
                    tmp3.append('공학')
                elif i == 1:
                    tmp3.append('인문/사회')
                elif i == 2:
                    tmp3.append('예체능')
                elif i == 3:
                    tmp3.append('교육')
                elif i == 4:
                    tmp3.append('자연')
                elif i == 5:
                    tmp3.append('의약')
        print(f'{exist_temp[0][0]}', tmp1)
        print(f'{exist_temp[1][0]}', tmp2)
        print(f'{exist_temp[2][0]}', tmp3)
        return self.preprocessingSimilarity(user_uid, exist_temp)