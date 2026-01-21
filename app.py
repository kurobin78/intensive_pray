import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import torch

@st.cache_resource
def load_all_bible():
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    
    # JSON 로드
    with open('bible_ko.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    bible_verses = data['verses'] # 구절 리스트 추출
        
    # 미리 계산된 임베딩 로드
    embeddings = np.load('bible_embeddings.npy')
    embeddings_tensor = torch.from_numpy(embeddings)
    
    return model, bible_verses, embeddings_tensor

model, bible_verses, bible_embeddings = load_all_bible()

st.title("🙏 마음을 만지는 말씀 분석기")

prayer_input = st.text_area("기도제목을 입력하세요", placeholder="예: 경제적인 문제로 마음이 무겁습니다.")

if st.button("말씀 찾기"):
    if prayer_input:
        query_embedding = model.encode(prayer_input, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, bible_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        
        result = bible_verses[best_idx]
        
        # 결과 출력 (책이름 장:절 형태)
        title = f"{result['book_name']} {result['chapter']}:{result['verse']}"
        content = result['text']
        
        st.success(f"### {title}")
        st.info(f"**{content}**")
    else:
        st.warning("기도제목을 입력해주세요.")

# --- 4. 하단 안내 ---

st.caption("© 2026 말씀의 등불 - 오픈소스 AI 모델을 사용하여 위로를 전합니다.")


