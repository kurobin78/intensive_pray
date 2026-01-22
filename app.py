import streamlit as st
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. 페이지 설정 및 보안 로드 ---
st.set_page_config(page_title="AI 기도 파트너", page_icon="🙏")

# Secrets 확인 및 Gemini 설정
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error("API 키 설정이 필요합니다. Streamlit Cloud의 Secrets 설정을 확인해주세요.")
    st.stop()

# --- 2. 이메일 발송 함수 ---
def send_email(receiver_email, content):
    try:
        sender_email = st.secrets["GMAIL_USER"]
        app_password = st.secrets["GMAIL_PASS"]
        
        msg = MIMEMultipart()
        msg['From'] = f"AI 기도 파트너 <{sender_email}>"
        msg['To'] = receiver_email
        msg['Subject'] = "[말씀의 등불] 당신을 위한 하나님의 말씀과 기도문입니다."
        
        msg.attach(MIMEText(content, 'plain'))
        
        # 보안을 위해 SMTP_SSL 또는 STARTTLS 사용
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"메일 발송 중 오류가 발생했습니다: {e}")
        return False

# --- 3. 웹 UI 디자인 ---
st.title("🙏 AI 기도 파트너")
st.markdown("당신의 고민과 기도제목을 들려주세요. Gemini AI가 말씀을 묵상하고 당신을 위한 기도문을 작성합니다.")

# 세션 상태 유지
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = None

prayer_topic = st.text_area("기도제목을 입력하세요", height=150, placeholder="예: 취업을 앞두고 결과가 두려워요. 위로를 얻고 싶습니다.")

if st.button("AI와 함께 기도 시작하기"):
    if prayer_topic.strip():
        with st.spinner('말씀을 묵상하고 기도문을 작성하고 있습니다...'):
            prompt = f"""
            당신은 따뜻하고 공감 능력이 뛰어난 기독교 상담가입니다. 
            사용자의 기도제목: "{prayer_topic}"
            
            위 내용을 바탕으로 다음을 작성해주세요:
            1. [마음 나누기]: 사용자의 슬픔이나 걱정에 공감하는 따뜻한 메시지
            2. [오늘의 말씀]: 상황에 가장 적합한 성경 구절 (장, 절 포함)
            3. [함께하는 기도]: 해당 말씀을 인용하여 사용자를 위해 정성스럽게 작성한 기도문
            
            형식: 정중하고 따뜻한 어조의 한국어. 마크다운 형식을 사용하여 가독성 좋게 출력할 것.
            """
            try:
                response = model.generate_content(prompt)
                st.session_state.ai_response = response.text
            except Exception as e:
                st.error(f"AI 응답 생성 실패: {e}")
    else:
        st.warning("기도제목을 입력해 주세요.")

# 결과 출력
if st.session_state.ai_response:
    st.markdown("---")
    st.markdown(st.session_state.ai_response)
    
    # 이메일 전송 섹션
    st.markdown("---")
    st.subheader("📬 이 내용을 이메일로 보내기")
    email_addr = st.text_input("받으실 이메일 주소", placeholder="example@gmail.com")
    
    if st.button("이메일 발송"):
        if email_addr:
            with st.spinner('이메일을 보내는 중...'):
                if send_email(email_addr, st.session_state.ai_response):
                    st.success(f"{email_addr}로 성공적으로 발송되었습니다!")
        else:
            st.warning("이메일 주소를 입력해주세요.")

st.caption("© 2026 AI Prayer Partner. Powered by Gemini 1.5 Flash.")
