import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 2. JSON 데이터 로드
with open('bible_ko.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터 구조에 맞게 'verses' 키에서 꺼내오기
bible_verses = data['verses']

# 'text' 키를 사용하여 본문 리스트 만들기
contents = [item['text'] for item in bible_verses]

print(f"총 {len(contents)}개의 구절을 분석합니다.")

# 3. 임베딩 생성 (약 5~10분 소요될 수 있음)
print("임베딩 시작... 잠시만 기다려 주세요.")
embeddings = model.encode(contents, show_progress_bar=True)

# 4. 결과 저장
np.save('bible_embeddings.npy', embeddings)
print("성공! 'bible_embeddings.npy' 파일이 생성되었습니다.")