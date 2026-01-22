import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. 이메일 발송 함수 ---
def send_email(receiver_email, prayer, verse_title, verse_content):
    try:
        sender_email = st.secrets["GMAIL_USER"]
        app_password = st.secrets["GMAIL_PASS"]
    except:
        st.error("Secrets 설정(GMAIL_USER, GMAIL_PASS)을 확인해주세요.")
        return False

    msg = MIMEMultipart()
    msg['From'] = f"말씀의 등불 <{sender_email}>"
    msg['To'] = receiver_email
    msg['Subject'] = "[말씀의 등불] 당신을 위한 하나님의 말씀이 도착했습니다."

    body = f"🙏 입력하신 기도제목:\n{prayer}\n\n📖 추천 말씀: {verse_title}\n{verse_content}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"메일 발송 실패: {e}")
        return False

# --- 2. 모델 및 5개로 쪼개진 데이터 로드 ---
@st.cache_resource
def load_resources():
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    
    with open('bible_ko.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    bible_verses = data['verses']
    
    # 5개의 조각을 순서대로 로드하여 합치기
    parts = []
    for i in range(1, 6):
        parts.append(np.load(f'embeddings_part{i}.npy'))
    
    combined = np.concatenate(parts)
    return model, bible_verses, torch.from_numpy(combined).float()

model, bible_verses, bible_embeddings = load_resources()

# --- 3. UI 및 검색 로직 ---
st.title("🙏 말씀의 등불 (Email Edition)")

if 'verse_result' not in st.session_state:
    st.session_state.verse_result = None

prayer_input = st.text_area("기도제목을 입력하세요", height=150)

if st.button("말씀 찾기"):
    if prayer_input.strip():
        with st.spinner('말씀을 찾는 중...'):
            q_emb = model.encode(prayer_input, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(q_emb, bible_embeddings)[0]
            best_idx = torch.argmax(scores).item()
            
            res = bible_verses[best_idx]
            st.session_state.verse_result = {
                'title': f"{res['book_name']} {res['chapter']}:{res['verse']}",
                'content': res['text'],
                'prayer': prayer_input
            }
    else:
        st.warning("내용을 입력해주세요.")

# 결과 출력 및 이메일 전송
if st.session_state.verse_result:
    v = st.session_state.verse_result
    st.success(f"### {v['title']}\n{v['content']}")
    
    st.divider()
    email_target = st.text_input("이 말씀을 받을 이메일 주소")
    if st.button("이메일로 보내기"):
        if email_target:
            if send_email(email_target, v['prayer'], v['title'], v['content']):
                st.success("메일 발송 성공!")
