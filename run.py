from d_knn import lectureKNN
from dataPreprocessing import DataPreprocessing


temp = DataPreprocessing()
knn = lectureKNN()

for i in range(20):
    tmp = knn.printClass(temp.matrix[f'user{i+1}'])
    exist_result = knn.existUser(f'user{i+1}')
    print(f'Target-User => user{i+1}', tmp)
    print(exist_result, '\n')

new = {'user101':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 5, 0, 0, 0]}
new_result = knn.newUser(new)
print('New-User => user101', knn.printClass(new['user101']))
print(new_result)