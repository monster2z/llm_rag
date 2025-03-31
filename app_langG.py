import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import time
import os
import tempfile
import pandas as pd
from typing import Annotated, Sequence, TypedDict, Union, List, Dict, Any
import operator
import json
from enum import Enum
import uuid

# LangChain 임포트 (최신 라이브러리 경로로 변경)
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    CSVLoader, 
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # Chroma 대신 FAISS 사용
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# LangGraph 임포트
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# langsmith로 로깅 설정
try:
    from langchain_teddynote import logging
    logging.langsmith("llm_rag_prototype")
except ImportError:
    print("langchain_teddynote 라이브러리를 설치하지 않았습니다. 로깅 기능이 비활성화됩니다.")

# SQLite 문제 해결을 위한 pysqlite3 설정
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    print("pysqlite3 라이브러리를 설치하지 않았습니다. sqlite3 관련 문제가 발생할 수 있습니다.")

# API 키 설정 (환경 변수에서 로드)
api_key = os.environ.get("OPENAI_API_KEY")
admin_pass = os.environ.get("ADMIN_PASS")
user_pass = os.environ.get("USER_PASS")
# anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")  # 필요시 활성화

# 임베딩 모델 선택
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM 모델 선택
LLM_MODEL = "gpt-4o-mini"  # 또는 "claude-3-sonnet-20240229"
LLM_PROVIDER = "openai"  # 또는 "anthropic"

# LangGraph 상태 정의
class AgentState(TypedDict):
    question: str
    context: List[str] 
    answer: str
    conversation_history: List[Dict[str, str]]
    sources: List[Dict[str, str]]
    need_more_info: bool
    username: str  # 사용자 식별을 위한 필드 추가

# 임시 데이터 저장 디렉토리
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# 페이지 설정
st.set_page_config(page_title="기업 내부용 LLM 프로토타입", layout="wide")

# 응답 생성 함수 - 함수 위치 이동
def generate_response(prompt, username, messages_key):
    # 대화 기록 가져오기
    conversation_history = [
        {"role": msg["role"], "content": msg["content"]} 
        for msg in st.session_state[messages_key]
    ]
    
    # LangGraph 워크플로우가 설정되어 있으면 해당 워크플로우 사용
    if 'rag_workflow' in st.session_state and 'vectorstore' in st.session_state:
        try:
            # 초기 상태 설정
            initial_state = {
                "question": prompt,
                "context": [],
                "answer": "",
                "conversation_history": conversation_history,
                "sources": [],
                "need_more_info": False,
                "username": username
            }
            
            # 워크플로우 실행
            result = st.session_state.rag_workflow.invoke(initial_state)
            
            # 답변 반환
            return result["answer"]
            
        except Exception as e:
            st.error(f"RAG 응답 생성 중 오류: {str(e)}")
            # 오류 발생 시 기본 응답으로 폴백
            return f"죄송합니다. 질문 처리 중 오류가 발생했습니다. 나중에 다시 시도해주세요."
    else:
        # 기본 LLM 사용 (RAG가 없는 경우)
        llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
            
        template = """
        당신은 기업 내부 AI 어시스턴트입니다. 
        사용자의 질문에 정확하게 답변하세요.
        
        현재 업로드된 문서가 없습니다. 일반적인 지식을 바탕으로 답변합니다.
        다만, 사용자에게 더 정확한 답변을 위해 관련 문서를 업로드하면 좋을 것이라고 알려주세요.
        
        이전 대화 기록: {conversation_history}
        
        질문: {question}
        
        답변:
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | llm | StrOutputParser()
        
        try:
            return chain.invoke({
                "question": prompt,
                "conversation_history": str(conversation_history)
            })
        except Exception as e:
            st.error(f"LLM 응답 생성 중 오류: {str(e)}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다. 나중에 다시 시도해주세요."

# LangGraph 노드 함수들

# 문서 검색 함수
def retrieve_documents(state: AgentState) -> AgentState:
    """문서 저장소에서 관련 문서를 검색하는 함수"""
    # 세션 상태에서 벡터 저장소 가져오기
    vectorstore = st.session_state.get("vectorstore")
    
    if not vectorstore:
        return {**state, "context": [], "sources": []}
    
    # 검색 수행
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(state["question"])
    
    # 컨텍스트 구성
    contexts = []
    sources = []
    
    for doc in docs:
        contexts.append(doc.page_content)
        sources.append({
            "source": doc.metadata.get("source_file", "Unknown"),
            "page": doc.metadata.get("page", "N/A")
        })
    
    return {**state, "context": contexts, "sources": sources}

# 질문 응답 생성 함수
def generate_answer(state: AgentState) -> AgentState:
    """검색된 문서를 바탕으로 질문에 대한 답변을 생성하는 함수"""
    # LLM 모델 초기화
    if LLM_PROVIDER == "anthropic":
        llm = ChatAnthropic(model=LLM_MODEL)
    else:
        llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
    
    # 프롬프트 템플릿 정의
    template = """
    당신은 기업 내부 문서에 대한 질문에 답변하는 AI 어시스턴트입니다.
    사용자의 질문에 대해 아래 문맥 정보를 참고하여 정확하게 답변하세요.
    문맥 정보에 답이 없는 경우, "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답하고 
    need_more_info를 True로 설정하세요. 그렇지 않으면 False로 설정하세요.
    
    이전 대화 기록: {conversation_history}
    
    문맥 정보:
    {context}
    
    질문: {question}
    
    답변:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 컨텍스트 구성
    context_text = "\n\n".join(state["context"]) if state["context"] else "관련 문서가 없습니다."
    
    # 이전 대화 기록
    conversation_history = state.get("conversation_history", [])
    
    # 입력 구성
    inputs = {
        "question": state["question"],
        "context": context_text,
        "conversation_history": str(conversation_history)
    }
    
    # 답변 생성
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(inputs)
    
    # 추가 정보 필요 여부 판단
    need_more_info = "제공된 문서에서 관련 정보를 찾을 수 없습니다" in answer
    
    return {
        **state, 
        "answer": answer, 
        "need_more_info": need_more_info
    }

# 소스 정보 추가 함수
def add_source_information(state: AgentState) -> AgentState:
    """답변에 소스 정보를 추가하는 함수"""
    if not state["sources"]:
        return state
        
    sources_info = "\n\n**참고 문서:**\n"
    for src in state["sources"]:
        sources_info += f"- {src['source']}"
        if src['page'] != "N/A":
            sources_info += f" (페이지: {src['page']})"
        sources_info += "\n"
    
    enhanced_answer = state["answer"] + sources_info
    
    return {**state, "answer": enhanced_answer}

# LangGraph 워크플로우 생성 함수
def create_rag_workflow():
    """RAG 워크플로우 생성"""
    # 워크플로우 정의
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("add_sources", add_source_information)
    
    # 엣지 설정
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "add_sources")
    workflow.add_edge("add_sources", END)
    
    # 시작 노드 설정
    workflow.set_entry_point("retrieve")
    
    # 그래프 컴파일
    return workflow.compile()

# 문서 처리 및 임베딩 함수
def process_documents(uploaded_files):
    documents = []
    file_info = []
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
            
            # 파일 정보 추가 (메타데이터)
            for doc in split_documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["source_file"] = uploaded_file.name
                doc.metadata["file_type"] = file_type
                # 업로더 정보 추가 (사용자별 문서 관리를 위해)
                doc.metadata["uploaded_by"] = st.session_state["username"]
            
            documents.extend(split_documents)
            file_info.append({
                "filename": uploaded_file.name,
                "file_type": file_type,
                "chunks": len(split_documents),
                "uploaded_by": st.session_state["username"],
                "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.sidebar.success(f"{uploaded_file.name} 처리 완료 - {len(split_documents)}개 청크 생성")
        except Exception as e:
            st.sidebar.error(f"{uploaded_file.name} 처리 중 오류 발생: {str(e)}")
        finally:
            # 임시 파일 삭제
            os.unlink(temp_file_path)
    
    # 임베딩 및 벡터 저장소 생성
    if documents:
        st.sidebar.info("문서 임베딩 중...")
        # 임베딩 모델 초기화
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
        
        # 기존 벡터스토어가 있으면 추가, 없으면 새로 생성
        if "vectorstore" in st.session_state:
            try:
                # 기존 벡터스토어에 문서 추가
                st.session_state.vectorstore.add_documents(documents)
                vectorstore = st.session_state.vectorstore
            except Exception as e:
                st.sidebar.warning(f"기존 벡터스토어에 추가 실패, 새로 생성합니다: {str(e)}")
                # FAISS 벡터스토어 생성
                vectorstore = FAISS.from_documents(documents, embeddings)
        else:
            # 새 FAISS 벡터스토어 생성
            vectorstore = FAISS.from_documents(documents, embeddings)
        
        # 로컬에 벡터스토어 저장 (나중에 로드할 수 있도록)
        vector_store_path = os.path.join(DATA_DIR, f"faiss_index_{uuid.uuid4().hex}")
        os.makedirs(vector_store_path, exist_ok=True)
        vectorstore.save_local(vector_store_path)
        
        st.sidebar.success(f"임베딩 완료! {len(documents)}개 문서 처리됨")
        
        # 파일 정보 저장
        if "uploaded_files_info" not in st.session_state:
            st.session_state.uploaded_files_info = []
        st.session_state.uploaded_files_info.extend(file_info)
        
        return vectorstore, file_info
    return None, []

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

# 사용자별 대화 기록 관리
def get_user_conversations(username):
    """사용자별 대화 목록 가져오기"""
    user_conv_key = f"conversations_{username}"
    if user_conv_key not in st.session_state:
        st.session_state[user_conv_key] = ["새 대화"]
    return st.session_state[user_conv_key]

def add_conversation(username, title=None):
    """사용자 대화 추가"""
    user_conv_key = f"conversations_{username}"
    if title is None:
        title = f"대화 {len(st.session_state[user_conv_key])}"
    st.session_state[user_conv_key].append(title)
    return len(st.session_state[user_conv_key]) - 1  # 새 대화 인덱스 반환

def get_conversation_messages(username, conv_index):
    """특정 대화의 메시지 가져오기"""
    messages_key = f"messages_{username}_{conv_index}"
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []
    return messages_key

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
    
    # 현재 사용자 정보 저장
    if "username" not in st.session_state:
        st.session_state["username"] = st.session_state["username"]  # 사용자 ID
        
    username = st.session_state["username"]  
    st.sidebar.title(f'{st.session_state["name"]} 님 환영합니다')
    
    # 사이드바 - 대화 목록 영역
    with st.sidebar:
        st.title("대화 목록")
        
        # 사용자별 대화 목록 관리
        conversations = get_user_conversations(username)
        
        # 현재 대화 상태 관리
        if f"current_conversation_{username}" not in st.session_state:
            st.session_state[f"current_conversation_{username}"] = 0
            
        current_conv_idx = st.session_state[f"current_conversation_{username}"]
        
        # 대화 목록 표시
        for i, conv in enumerate(conversations):
            if st.button(f"{conv}", key=f"conv_{i}"):
                st.session_state[f"current_conversation_{username}"] = i
                st.rerun()
        
        # 새 대화 버튼
        if st.button("새 대화 시작"):
            # 새 대화 추가
            new_idx = add_conversation(username)
            st.session_state[f"current_conversation_{username}"] = new_idx
            st.rerun()
            
        # 파일 업로드 섹션
        st.header("문서 업로드")
        uploaded_files = st.file_uploader("기업 내부 문서를 업로드하세요", 
                                       type=['pdf', 'docx', 'csv', 'pptx'], 
                                       accept_multiple_files=True)
        
        # 파일 처리 버튼
        if uploaded_files and st.button("문서 처리 및 임베딩"):
            with st.spinner("문서 처리 중..."):
                vectorstore, file_info = process_documents(uploaded_files)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    # LangGraph 워크플로우 생성
                    st.session_state.rag_workflow = create_rag_workflow()
                    st.success("문서가 성공적으로 처리되었습니다.")
                    
                    # 처리된 파일 정보 표시
                    if file_info:
                        st.subheader("처리된 파일")
                        for file in file_info:
                            st.write(f"📄 {file['filename']} - {file['chunks']}개 청크")
        
        # 업로드된 파일 목록 표시
        if "uploaded_files_info" in st.session_state and st.session_state.uploaded_files_info:
            st.header("업로드된 문서 목록")
            # 사용자별 파일 필터링
            user_files = [f for f in st.session_state.uploaded_files_info if f.get("uploaded_by") == username]
            
            if user_files:
                for file in user_files:
                    st.write(f"📄 {file['filename']} - {file['upload_time']}")
            else:
                st.write("업로드한 문서가 없습니다.")

    # 메인 컨테이너 - 채팅 영역
    st.title("기업 내부용 AI 어시스턴트")
    
    # 현재 대화 인덱스
    current_conv_idx = st.session_state[f"current_conversation_{username}"]
    
    # 현재 대화의 메시지 키
    current_messages_key = get_conversation_messages(username, current_conv_idx)
    
    # 메시지 표시
    for message in st.session_state[current_messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    prompt = st.chat_input("메시지를 입력하세요...")
    
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
                response = generate_response(prompt, username, current_messages_key)
            message_placeholder.markdown(response)
        
        # 어시스턴트 메시지 저장
        st.session_state[current_messages_key].append({"role": "assistant", "content": response})

elif st.session_state["authentication_status"] == False:
    st.error('사용자명/비밀번호가 올바르지 않습니다')
elif st.session_state["authentication_status"] is None:
    st.warning('사용자명과 비밀번호를 입력하세요')