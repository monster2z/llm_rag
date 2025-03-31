import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import time
import os

# 페이지 설정
st.set_page_config(page_title="기업 내부용 LLM 프로토타입", layout="wide")

# 사용자 인증 설정 파일 생성 함수
def create_config_file():
    # Hasher 객체 생성
    hasher = stauth.Hasher()
    
    config = {
        'credentials': {
            'usernames': {
                'admin': {
                    'email': 'admin@example.com',
                    'name': '관리자',
                    'password': hasher.hash('admin')
                },
                'user1': {
                    'email': 'user1@example.com',
                    'name': '사용자1',
                    'password': hasher.hash('password')
                }
            }
        },
        'cookie': {
            'expiry_days': 30,
            'key': 'some_signature_key',
            'name': 'llm_dashboard_cookie'
        }
    }
    
    with open('config.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    
    return config

# 설정 파일 로드 또는 생성
if not os.path.exists('config.yaml'):
    config = create_config_file()
else:
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=SafeLoader)

# 인증 객체 생성
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# 로그인 화면 - 0.4.2 버전에 맞게 수정
try:
    # 최신 버전 (0.4.x)의 login 메서드는 위치 인자가 필요 없음
    authenticator.login()
except Exception as e:
    st.error(f"로그인 처리 중 오류가 발생했습니다: {str(e)}")

# 로그인 상태 확인 및 메인 앱 표시
if st.session_state["authentication_status"]:
    # 로그인 성공 후 메인 화면
    authenticator.logout("로그아웃", "sidebar")
    st.sidebar.title(f'{st.session_state["name"]} 님 환영합니다')
    
    # 사이드바 - 대화 목록 영역
    with st.sidebar:
        st.title("대화 목록")
        if "conversations" not in st.session_state:
            st.session_state.conversations = ["새 대화"]
        if "current_conversation" not in st.session_state:
            st.session_state.current_conversation = 0
        
        for i, conv in enumerate(st.session_state.conversations):
            if st.button(f"{conv}", key=f"conv_{i}"):
                st.session_state.current_conversation = i
                if f"messages_{i}" not in st.session_state:
                    st.session_state[f"messages_{i}"] = []
                st.rerun()
        
        if st.button("새 대화 시작"):
            # 새 대화 추가
            new_idx = len(st.session_state.conversations)
            st.session_state.conversations.append(f"대화 {new_idx}")
            st.session_state.current_conversation = new_idx
            st.session_state[f"messages_{new_idx}"] = []
            st.rerun()  # 사이드바 업데이트를 위해 재실행
            
        # 파일 업로드 섹션
        st.header("문서 업로드")
        uploaded_file = st.file_uploader("기업 내부 문서를 업로드하세요", 
                                       type=['pdf', 'docx', 'csv', 'pptx'], 
                                       accept_multiple_files=True)

    # 메인 컨테이너 - 채팅 영역
    st.title("기업 내부용 AI 어시스턴트")

    # 현재 대화의 메시지 키
    current_messages_key = f"messages_{st.session_state.current_conversation}"
    
    # 메시지 기록 초기화
    if current_messages_key not in st.session_state:
        st.session_state[current_messages_key] = []

    # 메시지 표시
    for message in st.session_state[current_messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    prompt = st.chat_input("메시지를 입력하세요...")

    # LLM 응답 생성 함수 (실제 LLM 연동 시 수정 필요)
    def generate_response(prompt):
        # 이 부분은 나중에 실제 LLM API 연동으로 교체
        # 테스트용 간단한 응답
        time.sleep(1)  # 응답 대기시간 시뮬레이션
        return f"당신의 질문에 대한 응답입니다: {prompt}"

    # 입력 처리
    if prompt:
        # 사용자 메시지 추가
        st.session_state[current_messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 어시스턴트 응답
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = generate_response(prompt)
            message_placeholder.markdown(response)
        
        # 어시스턴트 메시지 저장
        st.session_state[current_messages_key].append({"role": "assistant", "content": response})

elif st.session_state["authentication_status"] == False:
    st.error('사용자명/비밀번호가 올바르지 않습니다')
elif st.session_state["authentication_status"] is None:
    st.warning('사용자명과 비밀번호를 입력하세요')