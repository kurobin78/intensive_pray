import streamlit as st
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 말씀 & 기도 파트너", page_icon="🙏")

# --- 2. [계정 A] Gemini API 설정 (최신 모델 적용) ---
try:
    gemini_key = st.secrets["ACCOUNTS"]["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_key)
    # 2026년 기준 최신 안정화 모델인 gemini-2.5-flash 사용
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error("Gemini API 설정을 확인해주세요. (모델명 또는 API 키 오류)")
    st.stop()

# --- 3. [계정 B] 이메일 발송 함수 ---
def send_email_via_account_b(receiver_email, content):
    try:
        sender_user = st.secrets["ACCOUNTS"]["GMAIL_USER"]
        sender_pass = st.secrets["ACCOUNTS"]["GMAIL_PASS"]
        
        msg = MIMEMultipart()
        msg['From'] = f"말씀의 등불 <{sender_user}>"
        msg['To'] = receiver_email
        msg['Subject'] = "[말씀의 등불] 당신을 위한 위로의 메시지와 기도문입니다."
        
        msg.attach(MIMEText(content, 'plain'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_user, sender_pass)
            server.sendmail(sender_user, receiver_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"메일 발송 실패: {e}")
        return False

# --- 4. 메인 UI ---
st.title("🙏 AI 말씀 & 기도 파트너")
st.markdown("당신의 마음을 남겨주세요. AI가 깊은 위로와 함께 기도문을 작성해 드립니다.")

if 'final_content' not in st.session_state:
    st.session_state.final_content = None

prayer_topic = st.text_area("기도제목이나 고민을 입력해 주세요", height=150)

if st.button("위로의 메세지 생성하기"):
    if prayer_topic.strip():
        with st.spinner('말씀을 묵상하며 기도문을 작성 중입니다...'):
            # 요청하신 순서와 구성을 반영한 프롬프트
            prompt = f"""
            사용자의 상황: "{prayer_topic}"
            
            위 내용을 바탕으로 다음 세 가지 섹션을 순서대로 작성해주세요. 
            어조는 매우 따뜻하고 깊은 울림이 있는 한국어 기독교 상담가 스타일이어야 합니다.

            1. [깊은 애도와 위로의 메시지]: 사용자의 아픔에 깊이 공감하고 위로하는 메시지를 작성하세요.
            2. [위로와 회복을 위한 기도문]: 사용자를 위한 간절한 기도문을 작성하되, 아래 3번에서 추천할 성경 말씀을 기도문 중간에 자연스럽게 인용하여 포함시키세요.
            3. [위로와 소망의 말씀]: 이 상황에 가장 힘이 되는 성경 구절(장, 절 포함)을 적어주세요.
            
            형식: 마크다운을 사용하여 각 섹션 제목을 명확히 구분하세요.
            """
            try:
                response = model.generate_content(prompt)
                st.session_state.final_content = response.text
            except Exception as e:
                st.error(f"AI 응답 생성 실패: {e}")
    else:
        st.warning("내용을 입력해주세요.")

# 결과 출력 및 이메일 전송
if st.session_state.final_content:
    st.markdown("---")
    st.markdown(st.session_state.final_content)
    
    st.divider()
    st.subheader("📬 이 내용을 이메일로 받기")
    target_email = st.text_input("수신 이메일 주소", placeholder="example@gmail.com")
    
    if st.button("이메일 발송"):
        if target_email:
            with st.spinner('메일을 보내는 중...'):
                if send_email_via_account_b(target_email, st.session_state.final_content):
                    st.success(f"{target_email}로 발송되었습니다!")
        else:
            st.error("이메일 주소를 입력해주세요.")
