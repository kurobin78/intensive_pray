mport streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. 이메일 발송 로직 ---
def send_email(receiver_email, prayer, verse_title, verse_content):
    # Streamlit Cloud의 Secrets 메뉴에 저장한 정보를 불러옵니다.
    try:
        sender_email = st.secrets["GMAIL_USER"]
        app_password = st.secrets["GMAIL_PASS"]
    except:
        st.error("이메일 설정을 찾을 수 없습니다. Secrets 설정을 확인하세요.")
        return False

    msg = MIMEMultipart()
    msg['From'] = f"말씀의 등불 <{sender_email}>"
    msg['To'] = receiver_email
    msg['Subject'] = "[말씀의 등불] 당신을 위한 하나님의 말씀이 도착했습니다."

    body = f"""
    당신의 기도를 응원합니다.
    
    🙏 입력하신 기도제목:
    {prayer}
    
    📖 당신을 위한 말씀: {verse_title}
    "{verse_content}"
    
    하나님의 평강이 당신의 마음과 생각을 지키시길 소망합니다.
    """
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"메일 발송 중 오류: {e}")
        return False

# --- 2. 모델 및 분할 데이터 로드 ---
@st.cache_resource
def load_resources():
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    
    with open('bible_ko.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    bible_verses = data['verses']
    
    # 쪼개진 두 파일을 로드하여 하나로 합치기
    p1 = np.load('embeddings_part1.npy')
    p2 = np.load('embeddings_part2.npy')
    combined = np.concatenate([p1, p2])
    
    return model, bible_verses, torch.from_numpy(combined).float()

model, bible_verses, bible_embeddings = load_resources()

# --- 3. UI 및 메인 로직 ---
st.title("🙏 말씀의 등불")
st.write("기도제목을 적어주시면 가장 가까운 말씀을 찾아 이메일로 보내드려요.")

# 세션 상태 초기화 (결과 유지용)
if 'found_verse' not in st.session_state:
    st.session_state.found_verse = None

prayer_input = st.text_area("당신의 기도를 적어주세요.", height=150)

if st.button("말씀 찾기"):
    if prayer_input.strip():
        with st.spinner('말씀을 묵상하며 찾는 중...'):
            q_emb = model.encode(prayer_input, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(q_emb, bible_embeddings)[0]
            best_idx = torch.argmax(scores).item()
            
            res = bible_verses[best_idx]
            st.session_state.found_verse = {
                'title': f"{res['book_name']} {res['chapter']}:{res['verse']}",
                'content': res['text'],
                'prayer': prayer_input
            }
    else:
        st.warning("기도제목을 입력해 주세요.")

# 결과 표시 및 이메일 폼
if st.session_state.found_verse:
    v = st.session_state.found_verse
    st.divider()
    st.success(f"### {v['title']}")
    st.info(f"**{v['content']}**")
    st.balloons()
    
    st.subheader("📬 이 말씀을 이메일로 받기")
    email = st.text_input("이메일 주소 입력")
    if st.button("메일 보내기"):
        if email:
            if send_email(email, v['prayer'], v['title'], v['content']):
                st.success("메일이 성공적으로 발송되었습니다!")
        else:
            st.error("이메일 주소를 입력해 주세요.")
