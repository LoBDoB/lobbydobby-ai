from d_knn import lectureKNN
from dataPreprocessing import DataPreprocessing


temp = DataPreprocessing()

knn = lectureKNN()

for i in range(20):
    tmp = []
    exist_result = knn.existUser(f'user{i+1}')
    for idx, t in enumerate(temp.matrix[f'user{i+1}'][-6:]):
        if t != 0:
            if idx == 0:
                tmp.append('공학')
            elif idx == 1:
                tmp.append('인문/사회')
            elif idx == 2:
                tmp.append('예체능')
            elif idx == 3:
                tmp.append('교육')
            elif idx == 4:
                tmp.append('자연')
            elif idx == 5:
                tmp.append('의약')
    print(f'Target-User => user{i+1}', tmp)
    print(exist_result, '\n')

new = {'user101':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 5, 0, 0, 0]}
new_result = knn.newUser(new)
print('\n', new_result)