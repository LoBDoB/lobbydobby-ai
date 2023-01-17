import numpy as np

matrix = np.array([[7,6,7,4,5,4],
                   [6,7,0,4,3,4],
                   [0,3,3,1,1,0],
                   [1,2,2,3,3,4],
                   [1,0,1,2,3,3]])

def cosine_simillarity(v1, v2):
  """
  두 벡터 v1, v2에 대한 코사인 유사도를 구하는 함수
  위 매트릭스에서 사용자 기반 추천을 한다고 할때, 사용자1을 v1, 사용자2를 v2로 놓는다면,
  v1 = [7,6,7,4,5]
  v2 = [6,7,?,4,3]
  으로 두고 함수를 적용하게 된다.

  return: similarity of the two vectors
  """
  A = np.sqrt(np.sum(np.square(v1)))
  B = np.sqrt(np.sum(np.square(v2)))
  return np.dot(v1,v2) / (A*B)

sim_lst = []  # 빈 리스트 생성
best_score = 0

for idx, vec in enumerate(matrix):  # 매트릭스의 각 사용자 별 벡터를 뽑아 vec에 넣기
  if idx != 2:
    similarity = cosine_simillarity(vec, matrix[2])  # matrix[2] == 사용자3의 벡터
    sim_lst.append((idx, similarity))
    if best_score < similarity:  # 현재 계산한 유사도가 기존 최고 유사도보다 높다면 바꿔준다.
      best_score = similarity
      best_user = idx +1

sim_lst.sort(key=lambda x: x[1], reverse=True)

print(sim_lst, f"\n사용자3과 가장 비슷한 유저: 사용자{best_user} \n유사도:{best_score}")