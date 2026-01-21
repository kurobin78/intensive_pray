import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 1. 성경 데이터 로드
with open('korean.json', 'r', encoding='utf-8') as f:
    bible_data = json.load(f)

contents = [item['content'] for item in bible_data]

# 2. 전체 구절 임베딩 (시간이 몇 분 소요될 수 있음)
print("임베딩 시작...")
bible_embeddings = model.encode(contents, show_progress_bar=True)

# 3. 결과 저장 (Numpy 배열로 저장)
np.save('bible_embeddings.npy', bible_embeddings)
print("저장 완료: bible_embeddings.npy")
