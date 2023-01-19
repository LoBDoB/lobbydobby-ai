import pandas as pd

class DataPreprocessing():
    def __init__(self):
        self.data = pd.read_csv('./data/knn_data.csv')
        self.matrix = {f'user{i + 1}': self.data.iloc[i].to_list() for i in range(len(self.data))}
        self.lectures = {0: '아픈 영혼을 철학으로 치유하기',
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