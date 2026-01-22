import streamlit as st
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. 페이지 설정 및 스타일 ---
st.set_page_config(page_title="AI 말씀 파트너", page_icon="🙏")

st.markdown("""
    <style>
    .result-box { padding: 20px; border-radius: 10px; background-color: #f9f9f9; border: 1px solid #eee; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. [계정 A] Gemini API 설정 ---
try:
    # 계정 A에서 발급받은 API 키 사용
    gemini_key = st.secrets["ACCOUNTS"]["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error("Gemini API 설정을 확인해주세요.")
    st.stop()

# --- 3. [계정 B] 이메일 발송 함수 ---
def send_email_via_account_b(receiver_email, content):
    try:
        # 계정 B (메일 전송 전용) 정보 사용
        sender_user = st.secrets["ACCOUNTS"]["GMAIL_USER"]
        sender_pass = st.secrets["ACCOUNTS"]["GMAIL_PASS"]
        
        msg = MIMEMultipart()
        msg['From'] = f"말씀의 등불 <{sender_user}>"
        msg['To'] = receiver_email
        msg['Subject'] = "[말씀의 등불] 당신을 위한 하나님의 말씀과 기도문"
        
        msg.attach(MIMEText(content, 'plain'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_user, sender_pass)
            server.sendmail(sender_user, receiver_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"메일 발송 실패 (계정 B 확인 필요): {e}")
        return False

# --- 4. 메인 UI 및 로직 ---
st.title("🙏 AI 말씀 & 기도 파트너")
st.info("AI 분석 계정과 메일 발송 계정이 분리되어 안전하게 운영됩니다.")

if 'final_content' not in st.session_state:
    st.session_state.final_content = None

prayer_topic = st.text_area("기도제목을 입력해 주세요", height=150)

if st.button("분석 및 기도문 생성"):
    if prayer_topic.strip():
        with st.spinner('Gemini AI(계정 A)가 말씀을 묵상 중입니다...'):
            prompt = f"기도제목: '{prayer_topic}'. 위로의 메시지, 성경구절, 기도문을 한국어로 정중하게 작성해줘."
            response = model.generate_content(prompt)
            st.session_state.final_content = response.text
    else:
        st.warning("내용을 입력해주세요.")

# 결과 노출 및 이메일 전송
if st.session_state.final_content:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown(st.session_state.final_content)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    st.subheader("📬 이 내용을 이메일로 받기 (Gmail 계정 B 사용)")
    target_email = st.text_input("수신 이메일 주소")
    
    if st.button("이메일 발송"):
        if target_email:
            with st.spinner('계정 B를 통해 메일을 보내는 중...'):
                if send_email_via_account_b(target_email, st.session_state.final_content):
                    st.success(f"{target_email}로 발송이 완료되었습니다!")
        else:
            st.warning("이메일 주소를 적어주세요.")


