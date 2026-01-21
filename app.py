import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

mport streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import torch

@st.cache_resource
def load_all_bible():
    # 모델 로드
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    
    # 성경 텍스트 로드
    with open('korean.json', 'r', encoding='utf-8') as f:
        bible_data = json.load(f)
        
    # 미리 계산된 임베딩 로드
    embeddings = np.load('bible_embeddings.npy')
    # 계산을 위해 다시 PyTorch 텐서로 변환
    embeddings_tensor = torch.from_numpy(embeddings)
    
    return model, bible_data, embeddings_tensor

model, bible_data, bible_embeddings = load_all_bible()

# --- 1. 페이지 스타일 및 설정 ---
st.set_page_config(page_title="말씀의 등불", page_icon="✨", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #4A90E2; color: white; }
    .verse-box { padding: 20px; border-radius: 15px; background-color: white; border-left: 5px solid #4A90E2; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 모델 및 데이터 로드 (캐싱) ---
@st.cache_resource
def load_resources():
    # 오픈소스 한국어 모델 로드
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    
    # 성경 데이터 (실제 서비스 시에는 더 많은 데이터를 추가하세요)
    bible_data = [
        {"verse": "빌립보서 4:6-7", "content": "아무 것도 염려하지 말고 다만 모든 일에 기도와 간구로, 너희 구할 것을 감사함으로 하나님께 아뢰라. 그리하면 모든 지각에 뛰어난 하나님의 평강이 그리스도 예수 안에서 너희 마음과 생각을 지키시리라."},
        {"verse": "시편 23:1", "content": "여호와는 나의 목자시니 내게 부족함이 없으리로다."},
        {"verse": "이사야 41:10", "content": "두려워하지 말라 내가 너와 함께 함이라 놀라지 말라 나는 네 하나님이 됨이라 내가 너를 굳세게 하리라 참으로 너를 도와 주리라 참으로 나의 의로운 오른손으로 너를 붙들리라."},
        {"verse": "마태복음 11:28", "content": "수고하고 무거운 짐 진 자들아 다 내게로 오라 내가 너희를 쉬게 하리라."},
        {"verse": "로마서 8:28", "content": "우리가 알거니와 하나님을 사랑하는 자 곧 그의 뜻대로 부르심을 입은 자들에게는 모든 것이 합력하여 선을 이루느니라."},
        {"verse": "여호수아 1:9", "content": "내가 네게 명령한 것이 아니냐 강하고 담대하라 두려워하지 말며 놀라지 말라 네가 어디로 가든지 네 하나님 여호와가 너와 함께 하느니라 하시니라."}
    ]
    contents = [d['content'] for d in bible_data]
    embeddings = model.encode(contents, convert_to_tensor=True)
    return model, bible_data, embeddings

model, bible_data, bible_embeddings = load_resources()

# --- 3. 웹 UI 구성 ---
st.title("✨ 말씀의 등불")
st.subheader("기도제목을 나누시면 위로의 말씀을 찾아드려요.")

with st.container():
    prayer_text = st.text_area("오늘의 기도제목이나 마음의 짐을 적어주세요.", 
                               placeholder="예: 취업 준비 중인데 자꾸 낙심이 되고 미래가 불안합니다...",
                               height=150)
    
    if st.button("하나님의 말씀 듣기"):
        if prayer_text.strip():
            with st.spinner('당신을 위한 말씀을 묵상하는 중...'):
                # 입력 문장 벡터화
                query_embedding = model.encode(prayer_text, convert_to_tensor=True)
                
                # 유사도 계산
                cos_scores = util.pytorch_cos_sim(query_embedding, bible_embeddings)[0]
                best_idx = torch.argmax(cos_scores).item()
                result = bible_data[best_idx]
                
                # 결과 표시
                st.write("---")
                st.markdown(f"""
                    <div class="verse-box">
                        <h3 style='color: #4A90E2;'>{result['verse']}</h3>
                        <p style='font-size: 1.2rem; line-height: 1.6;'>"{result['content']}"</p>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
        else:
            st.warning("기도제목을 먼저 입력해 주세요.")

# --- 4. 하단 안내 ---

st.caption("© 2026 말씀의 등불 - 오픈소스 AI 모델을 사용하여 위로를 전합니다.")
