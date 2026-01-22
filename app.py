import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. 이메일 발송 함수 (기존과 동일) ---
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

# --- 2. 모델 및 5개 분할 데이터 로드 ---
@st.cache_resource
def load_resources():
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    with open('bible_ko.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    bible_verses = data['verses']
    
    # 5개의 조각 로드
    parts = []
    for i in range(1, 6):
        parts.append(np.load(f'embeddings_part{i}.npy'))
    
    combined = np.concatenate(parts)
    return model, bible_verses, torch.from_numpy(combined).float()

model, bible_verses, bible_embeddings = load_resources()

# --- 3. UI 구성 ---
st.title("🙏 말씀의 등불")
st.write("기도제목을 적어주시면 가장 가까운 말씀을 찾아드립니다.")

if 'verse_result' not in st.session_state:
    st.session_state.verse_result = None

prayer_input = st.text_area("기도제목을 입력하세요", height=150)

if st.button("말씀 찾기"):
    if prayer_input.strip():
        with st.spinner('말씀을 묵상하며 찾는 중...'):
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

# --- 4. 결과 출력 (풍선 효과 삭제됨) ---
if st.session_state.verse_result:
    v = st.session_state.verse_result
    st.divider()
    
    # 풍선 효과(st.balloons)를 삭제하고 깔끔한 결과창만 남겼습니다.
    st.success(f"### {v['title']}")
    st.info(f"**{v['content']}**")
    
    st.divider()
    
    # 이메일 전송 UI
    st.subheader("📬 이 말씀을 이메일로 받기")
    email_target = st.text_input("이메일 주소 입력")
    if st.button("이메일로 보내기"):
        if email_target:
            with st.spinner('보내는 중...'):
                if send_email(email_target, v['prayer'], v['title'], v['content']):
                    st.success("메일이 발송되었습니다!")
        else:
            st.error("주소를 입력해 주세요.")
