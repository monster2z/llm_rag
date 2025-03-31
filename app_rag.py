import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import time
import os
import tempfile
import pandas as pd
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    CSVLoader, 
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
import langchain
from langchain_anthropic import ChatAnthropic

#.env 내용 로드 시킴
from dotenv import load_dotenv
import os
load_dotenv()

# langsmith로 로깅 설정
from langchain_teddynote import logging

logging.langsmith("llm_rag_prototype")

import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3

# OpenAI API 키 설정 (실제 사용시 환경변수 또는 안전한 방법으로 관리해야 함)
api_key = os.environ["OPENAI_API_KEY"]
admin_pass = os.environ.get("ADMIN_PASS")
user_pass = os.environ.get("USER_PASS")

# Anthropic API 키 설정 (미사용)
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

# 페이지 설정
st.set_page_config(page_title="기업 내부용 LLM 프로토타입", layout="wide")

# 사용자 인증 설정 파일 생성 함수
def create_config_file():
    hasher = stauth.Hasher()
    
    config = {
        'credentials': {
            'usernames': {
                'admin1': {
                    'email': 'admin@example.com',
                    'name': '관리자',
                    'password': hasher.hash(admin_pass)
                },
                'user17': {
                    'email': 'user1@example.com',
                    'name': '사용자1',
                    'password': hasher.hash(user_pass)
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

# 파일 타입에 따른 로더 선택 함수
def get_loader(file_path, file_type):
    if file_type == 'pdf':
        return PyPDFLoader(file_path)
    elif file_type == 'docx':
        return Docx2txtLoader(file_path)
    elif file_type == 'csv':
        return CSVLoader(file_path)
    elif file_type == 'pptx':
        return UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_type}")

# 문서 처리 및 임베딩 함수
def process_documents(uploaded_files):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    for uploaded_file in uploaded_files:
        # 파일 확장자 추출
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            # 로더 선택 및 문서 로드
            loader = get_loader(temp_file_path, file_type)
            loaded_documents = loader.load()
            
            # 문서 분할
            split_documents = text_splitter.split_documents(loaded_documents)
            documents.extend(split_documents)
            
            st.sidebar.success(f"{uploaded_file.name} 처리 완료")
        except Exception as e:
            st.sidebar.error(f"{uploaded_file.name} 처리 중 오류 발생: {str(e)}")
        finally:
            # 임시 파일 삭제
            os.unlink(temp_file_path)
    
    # 임베딩 및 벡터 저장소 생성
    if documents:
        st.sidebar.info("문서 임베딩 중...")
        # OpenAI 임베딩 사용 (또는 Anthropic 임베더로 대체 가능)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=api_key)
        # 임베딩 생성 및 Chroma DB에 저장
        vectorstore = FAISS.from_documents(documents, embeddings)
        st.sidebar.success("임베딩 완료!")
        return vectorstore
    return None

# RAG 파이프라인 생성 함수
def create_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # 프롬프트 템플릿 정의
    template = """
    당신은 기업 내부 문서에 대한 질문에 답변하는 AI 어시스턴트입니다.
    사용자의 질문에 대해 아래 문맥 정보를 참고하여 정확하게 답변하세요.
    문맥 정보에 답이 없는 경우, "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답하세요.
    
    문맥 정보:
    {context}
    
    질문: {question}
    
    답변:
    
    관련 근거: 
    
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # LLM 선택 (OpenAI 또는 Anthropic)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)  # OpenAI 모델
    # llm = ChatAnthropic(model="claude-3-sonnet-20240229")  # Anthropic 모델
    
    # RAG 파이프라인 정의
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

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

# 로그인 화면 - 0.4.x 버전에 맞춤
authenticator.login()

# 로그인 상태에 따른 화면 전환
if st.session_state["authentication_status"]:
    # 로그인 성공 후 메인 화면
    authenticator.logout('로그아웃', 'sidebar')
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
            new_conv_idx = len(st.session_state.conversations)
            st.session_state.conversations.append(f"대화 {new_conv_idx}")
            st.session_state.current_conversation = new_conv_idx
            st.session_state[f"messages_{new_conv_idx}"] = []
            st.rerun()  # 사이드바 업데이트를 위해 재실행
            
        # 파일 업로드 섹션
        st.header("문서 업로드")
        uploaded_files = st.file_uploader("기업 내부 문서를 업로드하세요", 
                                       type=['pdf', 'docx', 'csv', 'pptx'], 
                                       accept_multiple_files=True)
        
        # 파일 처리 버튼
        if uploaded_files and st.button("문서 처리 및 임베딩"):
            with st.spinner("문서 처리 중..."):
                vectorstore = process_documents(uploaded_files)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.rag_pipeline = create_rag_pipeline(vectorstore)
                    st.success("문서가 성공적으로 처리되었습니다.")

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

    # 응답 생성 함수
    def generate_response(prompt):
        # RAG 파이프라인이 설정되어 있으면 해당 파이프라인 사용
        if 'rag_pipeline' in st.session_state and 'vectorstore' in st.session_state:
            try:
                # RAG를 사용한 응답 생성
                return st.session_state.rag_pipeline.invoke(prompt)
            except Exception as e:
                st.error(f"RAG 응답 생성 중 오류: {str(e)}")
                # 오류 발생 시 기본 응답으로 폴백
                return f"죄송합니다. 질문 처리 중 오류가 발생했습니다. 나중에 다시 시도해주세요."
        else:
            # RAG가 설정되지 않은 경우 기본 응답
            # 이 부분은 실제 LLM API 연동으로 교체
            time.sleep(1)  # 응답 대기시간 시뮬레이션
            return f"현재 업로드된 문서가 없습니다. 문서를 업로드하여 더 정확한 답변을 받을 수 있습니다.\n\n일반적인 응답: {prompt}"

    # 입력 처리
    if prompt:
        # 사용자 메시지 추가
        st.session_state[current_messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 어시스턴트 응답
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("응답 생성 중..."):
                response = generate_response(prompt)
            message_placeholder.markdown(response)
        
        # 어시스턴트 메시지 저장
        st.session_state[current_messages_key].append({"role": "assistant", "content": response})

elif st.session_state["authentication_status"] == False:
    st.error('사용자명/비밀번호가 올바르지 않습니다')
elif st.session_state["authentication_status"] is None:
    st.warning('사용자명과 비밀번호를 입력하세요')