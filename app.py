import streamlit as st
from streamlit_geolocation import streamlit_geolocation
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import os
import re
import glob
import io

# Langchain 관련 import
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# .env 파일 로드 (로컬 개발 시 사용. Streamlit Cloud에서는 Secrets 사용 권장)
load_dotenv()

st.set_page_config(page_title="🚂관광지 추천 챗봇", layout="wide")

# --- 커스텀 CSS 정의 ---
# 여기서 불필요하거나 잘못된 문자를 제거했습니다.
st.markdown(
    """
    <style>
    /* 전체 배경색 및 폰트 */
    .stApp {
        background-color: #f8f9fa; /* 밝은 회색, 거의 흰색 */
        color: #343a40; /* 어두운 회색 텍스트 */
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* 제목 스타일 */
    h1 {
        color: #007bff; /* 강렬한 파란색 */
        text-align: center;
        font-size: 3.2em;
        margin-bottom: 0.6em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h2 {
        color: #28a745; /* 초록색 강조 */
        font-size: 2.2em;
        border-bottom: 3px solid #e9ecef; /* 깔끔한 구분선 */
        padding-bottom: 0.4em;
        margin-top: 2em;
        margin-bottom: 1.5em;
        display: flex;
        align-items: center;
    }
    h2 .icon {
        font-size: 1.2em;
        margin-right: 10px;
    }
    h3 {
        color: #6c757d; /* 중간 회색 */
        font-size: 1.6em;
        margin-top: 1.5em;
        margin-bottom: 1em;
    }
    h4 {
        color: #495057;
        font-size: 1.2em;
        margin-top: 1em;
        margin-bottom: 0.6em;
    }

    /* 사이드바 스타일 */
    .stSidebar {
        background-color: #ffffff; /* 흰색 사이드바 */
        color: #343a40;
        border-right: 1px solid #dee2e6;
        box-shadow: 2px 0 8px rgba(0,0,0,0.05);
    }
    .stSidebar .stButton>button {
        width: 100%;
        margin-bottom: 8px;
        border-radius: 8px;
        border: none;
        background-color: #e9ecef; /* 버튼 배경색 */
        color: #343a40;
        font-size: 1em;
        padding: 10px 15px;
        transition: all 0.2s ease-in-out;
    }
    .stSidebar .stButton>button:hover {
        background-color: #007bff; /* 호버시 색상 */
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stSidebar .stSubheader {
        color: #007bff;
        text-align: center;
        margin-bottom: 1.5em;
        font-size: 1.4em;
    }
    .stSidebar .stInfo {
        background-color: #e0f7fa;
        border-left: 5px solid #00acc1;
        padding: 10px;
        border-radius: 8px;
        margin-top: 15px;
    }

    /* 입력 위젯 스타일 */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div, .stMultiSelect>div>div>div {
        border-radius: 10px;
        border: 1px solid #ced4da;
        padding: 12px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        background-color: #ffffff;
        font-size: 1.05em;
    }
    .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #ced4da;
        padding: 12px;
        background-color: #ffffff;
        font-size: 1.05em;
    }
    .stForm {
        padding: 30px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 30px;
    }
    .stForm button {
        margin-top: 20px;
    }

    /* 버튼 스타일 */
    .stButton>button {
        background-color: #007bff; /* 주 버튼 파란색 */
        color: white;
        border-radius: 12px;
        padding: 12px 25px;
        font-size: 1.2em;
        font-weight: bold;
        border: none;
        box-shadow: 0 5px 10px rgba(0,123,255,0.2);
        transition: all 0.3s ease-in-out;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3; /* 호버 시 더 진하게 */
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0,123,255,0.3);
    }
    /* 특정 버튼 (새로운 대화 시작하기) 스타일 */
    /* Streamlit 1.28+에서 button key에 따라 style 주는 방법 */
    /* 현재 코드에서는 st.button("새로운 대화 시작하기")에 kind="secondary"를 명시적으로 주지 않았으므로 이 선택자는 적용되지 않을 수 있습니다. */
    /* 만약 특정 버튼에 스타일을 적용하고 싶다면 st.button("새로운 대화 시작하기", type="secondary")와 같이 type을 지정하거나, 
        버튼의 key를 활용하는 다른 CSS 선택자를 고려해야 합니다. */
    .stButton button[kind="secondary"] { 
        background-color: #6c757d; /* 회색 버튼 */
        box-shadow: 0 3px 6px rgba(108,117,125,0.2);
    }
    .stButton button[kind="secondary"]:hover {
        background-color: #5a6268;
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(108,117,125,0.3);
    }


    /* 경고/성공/정보 메시지 스타일 */
    .stAlert {
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-size: 1.05em;
    }
    .stAlert.success {
        background-color: #d4edda;
        color: #155724;
        border-left: 5px solid #28a745;
    }
    .stAlert.warning {
        background-color: #fff3cd;
        color: #856404;
        border-left: 5px solid #ffc107;
    }
    .stAlert.error {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 5px solid #dc3545;
    }
    .stAlert.info {
        background-color: #e0f7fa;
        color: #0056b3;
        border-left: 5px solid #007bff;
    }
    
    /* 스피너 스타일 */
    .stSpinner > div > div {
        color: #007bff; /* 스피너 색상 변경 */
    }

    /* 마크다운 테이블 스타일 (여행 계획표) */
    /* Streamlit의 st.dataframe은 AgGrid 기반이므로 AgGrid 관련 클래스를 사용합니다. */
    /* 아래 CSS는 st.dataframe에만 적용될 수 있습니다. st.markdown으로 생성된 테이블에는 적용되지 않을 수 있습니다. */
    .st-ag .ag-header-cell {
        background-color: #007bff !important;
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 1.1em;
    }
    .st-ag .ag-cell {
        background-color: #ffffff !important;
        color: #343a40 !important;
        padding: 12px !important;
    }
    .st-ag .ag-row-even {
        background-color: #f8f9fa !important;
    }
    .st-ag .ag-row-odd {
        background-color: #ffffff !important;
    }
    .ag-root-wrapper {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    /* 입력 필드 레이블 */
    .stTextInput label, .stTextArea label, .stSelectbox label, .stMultiSelect label, .stNumberInput label {
        font-weight: bold;
        color: #495057;
        font-size: 1.1em;
        margin-bottom: 0.5em;
    }

    /* 구분선 */
    hr {
        margin-top: 3em;
        margin-bottom: 3em;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 123, 255, 0.75), rgba(0, 0, 0, 0));
    }

    /* selectbox 텍스트 잘림 해결 */
    .stSelectbox>div>div>div>div {
        width: 100% !important; /* 부모 너비에 맞춤 */
        white-space: normal !important; /* 텍스트 줄바꿈 허용 */
        overflow-wrap: break-word !important; /* 긴 단어 강제 줄바꿈 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 파일 경로 정의 (상수) ---
# GitHub 저장소에 업로드할 때 이 경로가 올바르게 설정되어 있어야 합니다.
# 예: 프로젝트 루트에 CSV 파일들이 있다면 "./파일명.csv"
VECTOR_DB_PATH = "faiss_tourist_attractions"

# 로드할 개별 관광지 CSV 파일 목록을 직접 지정합니다.
# 이 파일들은 GitHub 저장소의 앱 스크립트와 동일한 위치 또는 지정된 상대 경로에 있어야 합니다.
TOUR_CSV_FILES = [
    "./경기도역사관광지현황.csv",
    "./경기도자연관광지현황.csv",
    "./경기도체험관광지현황.csv",
    "./경기도테마관광지현황.csv",
    "./관광지정보현황(제공표준).csv",
    "./관광지현황.csv",
    # 필요에 따라 다른 CSV 파일들을 여기에 추가하세요.
]

# --- 초기 파일 존재 여부 확인 ---
required_files = TOUR_CSV_FILES
for f_path in required_files:
    # GitHub 배포 시, 이 os.path.exists 검사는 Git 저장소 내의 파일 존재 여부를 확인합니다.
    if not os.path.exists(f_path):
        st.error(f"필수 데이터 파일 '{f_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요. (Streamlit Cloud에서는 해당 파일들이 Git 리포지토리에 포함되어야 합니다.)")
        st.stop()


# --- 1. 설정 및 초기화 함수 ---
def setup_environment():
    """
    환경 변수 또는 Streamlit secrets에서 OpenAI API 키를 로드합니다.
    Streamlit Cloud 환경에서는 st.secrets를 우선적으로 사용합니다.
    로컬 환경에서는 .env 파일을 로드하거나 시스템 환경 변수에서 가져옵니다.
    """
    if 'OPENAI_API_KEY' in st.secrets:
        return st.secrets['OPENAI_API_KEY']
    else:
        # load_dotenv()는 이 함수 바깥에서 한 번 호출되므로 여기서는 생략
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            pass
        else:
            st.error("❌ OpenAI API 키를 찾을 수 없습니다. Streamlit Cloud에서는 secrets.toml에 키를 설정하거나, 로컬에서는 .env 파일을 확인해주세요.")
        return api_key

# --- 2. 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_specific_tour_data(file_paths_list):
    """지정된 CSV 파일 목록을 로드하고, 모든 파일에 CP949 인코딩을 적용하여 병합합니다."""
    combined_df = pd.DataFrame()

    if not file_paths_list:
        st.error("로드할 관광지 CSV 파일 경로가 지정되지 않았습니다. `TOUR_CSV_FILES`를 확인해주세요.")
        st.stop()

    for file_path in file_paths_list:
        if not os.path.exists(file_path):
            st.warning(f"'{file_path}' 파일을 찾을 수 없어 건너뜱니다. (Streamlit Cloud에서는 해당 파일들이 Git 리포지토리에 포함되어야 합니다.)")
            continue

        # 'cp494' 오류 수정: 'cp949'로 변경
        current_encoding = 'cp949'  

        try:
            # GitHub에 파일이 있다면, Streamlit은 해당 경로에서 파일을 읽어옵니다.
            df = pd.read_csv(file_path, encoding=current_encoding)
            df.columns = df.columns.str.strip()

            if "위도" not in df.columns or "경도" not in df.columns:
                st.warning(f"'{os.path.basename(file_path)}' 파일은 '위도', '경도' 컬럼이 없어 건너뜱니다.")
                continue

            name_col = None
            for candidate in ["관광지명", "관광정보명","관광지"]:
                if candidate in df.columns:
                    name_col = candidate
                    break

            if name_col is None:
                df["관광지명"] = "이름 없음"
            else:
                df["관광지명"] = df[name_col]

            address_col = None
            for candidate in ["정제도로명주소","정제지번주소","소재지도로명주소","소재지지번주소","관광지소재지지번주소","관광지소재지도로명주소"]:
                if candidate in df.columns:
                    address_col = candidate
                    break

            if address_col is None:
                df["소재지도로명주소"] = "주소 없음"
            else:
                df["소재지도로명주소"] = df[address_col]

            df = df[["위도", "경도", "관광지명", "소재지도로명주소"]]

            combined_df = pd.concat([combined_df, df], ignore_index=True)

        except Exception as e:
            st.warning(f"'{os.path.basename(file_path)}' 파일 ({current_encoding} 인코딩 시도) 처리 중 오류 발생: {e}")

    if combined_df.empty:
        st.error("지정된 파일들에서 유효한 관광지 데이터를 불러오지 못했습니다. `TOUR_CSV_FILES`와 파일 내용을 확인해주세요.")
        st.stop()

    return combined_df


# --- 벡터스토어 로딩 및 캐싱 ---
@st.cache_resource
def load_and_create_vectorstore_from_specific_files(tour_csv_files_list):
    """지정된 CSV 파일 목록을 사용하여 벡터스토어를 생성합니다."""
    all_city_tour_docs = []
    for file_path in tour_csv_files_list:
        if not os.path.exists(file_path):
            st.warning(f"벡터스토어 생성을 위해 '{file_path}' 파일을 찾을 수 없어 건너뜱니다.")
            continue

        current_encoding = 'cp949'  

        try:
            city_tour_loader = CSVLoader(file_path=file_path, encoding=current_encoding, csv_args={'delimiter': ','})
            all_city_tour_docs.extend(city_tour_loader.load())
        except Exception as e:
            st.warning(f"'{os.path.basename(file_path)}' 파일 ({current_encoding} 인코딩 시도) 로드 중 오류 발생 (벡터스토어): {e}")

    all_documents = all_city_tour_docs

    if not all_documents:
        st.error("벡터스토어를 생성할 문서가 없습니다. CSV 파일 경로와 내용을 확인해주세요.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    docs = text_splitter.split_documents(all_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    return vectorstore

@st.cache_resource()
def get_vectorstore_cached(tour_csv_files_list):
    """캐시된 벡터스토어를 로드하거나 새로 생성합니다."""
    if os.path.exists(VECTOR_DB_PATH):
        try:
            return FAISS.load_local(
                VECTOR_DB_PATH,
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"기존 벡터 DB 로딩 실패: {e}. 새로 생성합니다.")
            return load_and_create_vectorstore_from_specific_files(tour_csv_files_list)
    else:
        return load_and_create_vectorstore_from_specific_files(tour_csv_files_list)


# --- Haversine distance function ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# --- 3. 사용자 입력 및 UI 로직 함수 ---
def get_user_inputs_ui():
    """사용자로부터 나이, 여행 스타일, 현재 위치, 그리고 추가 여행 계획 정보를 입력받는 UI를 표시합니다."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("#### 사용자 정보 입력")
        # '나이대 선택' selectbox의 너비를 CSS로 조절하기 위해 key를 부여
        age = st.selectbox("나이대 선택", ["10대", "20대", "30대", "40대", "50대 이상"], key='age_selectbox_new')
        travel_style = st.multiselect("여행 스타일", ["자연", "역사", "체험", "휴식", "문화", "가족", "액티비티"], key='travel_style_multiselect')

    st.header("① 위치 가져오기")
    location = streamlit_geolocation()

    user_lat_final, user_lon_final = None, None

    if location and "latitude" in location and "longitude" in location:
        temp_lat = location.get("latitude")
        temp_lon = location.get("longitude")
        if temp_lat is not None and temp_lon is not None:
            user_lat_final = temp_lat
            user_lon_final = temp_lon
            st.success(f"📍 현재 위치: 위도 {user_lat_final:.7f}, 경도 {user_lon_final:.7f}")
        else:
            st.warning("📍 위치 정보를 불러오지 못했습니다. 수동으로 입력해 주세요.")
    else:
        st.warning("위치 정보를 사용할 수 없습니다. 수동으로 위도, 경도를 입력해 주세요.")

    if user_lat_final is None or user_lon_final is None:
        default_lat = st.session_state.get("user_lat", 37.5665) # 서울 시청 기본 위도
        default_lon = st.session_state.get("user_lon", 126.9780) # 서울 시청 기본 경도

        st.subheader("직접 위치 입력 (선택 사항)")
        manual_lat = st.number_input("위도", value=float(default_lat), format="%.7f", key="manual_lat_input")
        manual_lon = st.number_input("경도", value=float(default_lon), format="%.7f", key="manual_lon_input")

        if manual_lat != 0.0 or manual_lon != 0.0:
            user_lat_final = manual_lat
            user_lon_final = manual_lon
        else:
            user_lat_final = None
            user_lon_final = None
            st.error("유효한 위도 및 경도 값이 입력되지 않았습니다. 0이 아닌 값을 입력해주세요.")

    st.session_state.user_lat = user_lat_final
    st.session_state.user_lon = user_lon_final

    st.markdown("#### 추가 여행 계획 정보")
    trip_duration_days = st.number_input("여행 기간 (일)", min_value=1, value=3, key='trip_duration')
    estimated_budget = st.number_input("예상 예산 (원, 총 금액)", min_value=0, value=500000, step=10000, key='estimated_budget')
    num_travelers = st.number_input("여행 인원 (명)", min_value=1, value=2, key='num_travelers')
    special_requests = st.text_area("특별히 고려할 사항 (선택 사항)", help="예: 유모차 사용, 고령자 동반, 특정 음식 선호 등", key='special_requests')

    return age, travel_style, user_lat_final, user_lon_final, trip_duration_days, estimated_budget, num_travelers, special_requests

# --- 4. 추천 로직 함수 (Langchain API 변경: create_retrieval_chain 사용) (프롬프트 수정) ---
@st.cache_resource
def get_qa_chain(_vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    qa_prompt = PromptTemplate.from_template(
        """
당신은 사용자 위치 기반 여행지 추천 및 상세 여행 계획 수립 챗봇입니다.
사용자의 나이대, 여행 성향, 현재 위치 정보, 그리고 다음의 추가 정보를 참고하여 사용자가 입력한 질문에 가장 적합한 관광지를 추천하고, 이를 바탕으로 상세한 여행 계획을 수립해 주세요.
**관광지 추천 시 사용자 위치로부터의 거리는 시스템이 자동으로 계산하여 추가할 것이므로, 답변에서 거리를 직접 언급하지 마십시오.**
특히, 사용자의 현재 위치({user_lat}, {user_lon})에서 가까운 장소들을 우선적으로 고려하여 추천하고 계획을 세워주세요.
꼭꼭 사용자 현재 위치와 가까운 곳을 최우선으로 해주고 사용자가 선택한 성향에 맞게 추천해주세요.

[관광지 데이터]
{context}

[사용자 정보]
나이대: {age}
여행 성향: {travel_style}
현재 위치 (위도, 경도): {user_lat}, {user_lon}
여행 기간: {trip_duration_days}일
예상 예산: {estimated_budget}원
여행 인원: {num_travelers}명
특별 고려사항: {special_requests}

[사용자 질문]
{input}

다음 지침에 따라 상세한 여행 계획을 세워주세요:
1.  **관광지 추천:** 질문에 부합하고, 사용자 위치에서 가까운 1~3개의 주요 관광지를 추천하고, 각 관광지에 대한 다음 정보를 제공하세요.
    * 관광지 이름: [관광지명]
    * 주소: [주소]
    * 주요 시설/특징: [정보]
    **[참고: 사용자 위치 기준 거리는 시스템이 자동으로 계산하여 추가할 것이므로, 이 항목은 제외합니다.]**
    
2.  **추천된 관광지를 포함하여, 사용자 정보와 질문에 기반한 {trip_duration_days}일간의 상세 여행 계획을 일자별로 구성해 주세요.**
    * 각 날짜별로 방문할 장소(식당, 카페, 기타 활동 포함), 예상 시간, 간단한 활동 내용을 포함하세요.
    * 예산을 고려하여 적절한 식사 장소나 활동을 제안할 수 있습니다.
    * 이동 경로(예: "도보 15분", "버스 30분")를 간략하게 언급해 주세요.
    * 계획은 명확하고 이해하기 쉽게 작성되어야 합니다.

[답변 예시]
**추천 관광지:**
- 관광지 이름: [관광지명 1]
  - 주소: [주소 1]
  - 주요 시설/특징: [정보 1]
- 관광지 이름: [관광지명 2]
  - 주소: [주소 2]
  - 주요 시설/특징: [정보 2]

**상세 여행 계획 ({trip_duration_days}일):**
다음 표 형식으로 일자별 상세 계획을 작성해 주세요. 컬럼명은 '일차', '시간', '활동', '예상 장소', '이동 방법'으로 해주세요.
| 일차 | 시간 | 활동 | 예상 장소 | 이동 방법 |
|---|---|---|---|---|
| 1일차 | 오전 (9:00 - 12:00) | [활동 내용] | [장소명] | [이동 방법] |
| 1일차 | 점심 (12:00 - 13:00) | [식사] | [식당명] | - |
| 1일차 | 오후 (13:00 - 17:00) | [활동 내용] | [장소명] | [이동 방법] |
| 1일차 | 저녁 (17:00 이후) | [활동 내용] | [장소명 또는 자유 시간] | - |
| 2일차 | ... | ... | ... | ... |
**중요: '일차' 컬럼의 경우, 같은 일차의 여러 활동이 있을 경우 첫 번째 활동에만 해당 '일차'를 명시하고, 나머지 활동 행의 '일차' 셀은 비워두세요 (예: "| | 시간 | 활동 | 예상 장소 | 이동 방법 |"). 이렇게 해야 표에서 '일차'가 자동으로 병합되어 보입니다.**
"""
    )
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 15})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


# --- 5. 메인 앱 실행 로직 ---
if __name__ == "__main__":
    openai_api_key = setup_environment()
    if not openai_api_key:
        st.stop()
    
    # 세션 상태 초기화: 'app_started' 플래그 추가
    if "app_started" not in st.session_state:
        st.session_state.app_started = False
    
    # 세션 상태 초기화 및 이전 대화 기록 관리
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
        st.session_state.current_input = ""
        st.session_state.selected_conversation_index = None
    if "messages" in st.session_state: # 기존 messages 상태가 남아있을 경우 삭제
        del st.session_state.messages

    # 시작 화면
    if not st.session_state.app_started:
        st.title("🚂떠나자! 맞춤형 여행 계획 챗봇")
        st.markdown("### 당신의 완벽한 여행을 위한 AI 파트너")
        
        # PNG 이미지 파일 경로 (예: airplane.png)
        local_image_path = "./train.jpg"  
        
        # 이미지 파일 존재 여부 확인 (GitHub 배포 시 경로 확인에 유용)
        if os.path.exists(local_image_path):
            st.image(local_image_path,  
                     caption="여행의 시작은 지금부터!",  
                     use_container_width=True)  
        else:
            # 이미지가 없을 경우 대체 텍스트 또는 경고 메시지 표시
            st.warning(f"시작 화면 이미지를 찾을 수 없습니다: {local_image_path}") #
            # 또는 대체 URL 이미지를 사용할 수도 있습니다.
            # st.image("https://images.unsplash.com/photo-1542171124-ed989b5c3ee5?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",  
            #          caption="여행의 시작은 비행기에서부터!",  
            #          use_container_width=True)

        st.write("""
        이 챗봇은 당신의 나이대, 여행 스타일, 현재 위치를 기반으로 최적의 관광지를 추천하고,  
        상세한 일자별 여행 계획을 세워줍니다.  
        이제 번거로운 계획은 AI에게 맡기고 즐거운 여행만 준비하세요!
        """)
        
        if st.button("🚂여행 계획 시작하기"):
            st.session_state.app_started = True
            st.rerun() # 앱 다시 시작하여 챗봇 화면으로 전환

    else: # 앱 시작 플래그가 True인 경우 챗봇 화면 표시
        st.title("🗺️ 위치 기반 관광지 추천 및 여행 계획 챗봇")
        vectorstore = get_vectorstore_cached(TOUR_CSV_FILES)
        tour_data_df = load_specific_tour_data(TOUR_CSV_FILES)
        qa_chain = get_qa_chain(vectorstore) # DataFrame 로드 후 qa_chain 초기화

        # Sidebar for previous conversations
        with st.sidebar:
            st.subheader("💡이전 대화")
            if st.session_state.conversations:
                # 최신 대화를 먼저 보여주기 위해 역순으로 반복
                for i, conv in enumerate(reversed(st.session_state.conversations)):
                    original_index = len(st.session_state.conversations) - 1 - i
                    
                    if 'travel_style_selected' in conv and conv['travel_style_selected'] and conv['travel_style_selected'] != '특정 없음':
                        preview_text = f"성향: {conv['travel_style_selected']}"
                        # 미리보기 텍스트가 너무 길면 잘라냄
                        if len(preview_text) > 25:
                            preview_text = preview_text[:22] + '...'
                    else:
                        preview_text = conv['user_query'][:25] + ('...' if len(conv['user_query']) > 25 else '')
                        
                    if st.button(f"대화 {original_index + 1}: {preview_text}", key=f"sidebar_conv_{original_index}"):
                        st.session_state.selected_conversation_index = original_index
                        st.rerun()

            else:
                st.info("이전 대화가 없습니다.")

        # --- 메인 콘텐츠 영역 ---
        if st.session_state.selected_conversation_index is not None:
            st.header("이전 대화 내용")
            
            selected_conv = st.session_state.conversations[st.session_state.selected_conversation_index]
            
            st.subheader("질문:")
            st.markdown(selected_conv['user_query'])
            
            if 'travel_style_selected' in selected_conv and selected_conv['travel_style_selected'] and selected_conv['travel_style_selected'] != '특정 없음':
                st.subheader("성향:")
                st.markdown(selected_conv['travel_style_selected'])

            st.subheader("답변:")
            st.markdown(selected_conv['chatbot_response']) # 원본 텍스트로 보여줌
            
            st.markdown("---")
            if st.button("새로운 대화 시작하기"):
                st.session_state.selected_conversation_index = None
                st.session_state.current_input = ""
                st.rerun()

        else: # 이전 대화가 선택되지 않은 경우 (새로운 질문 입력 상태)
            age, travel_style_list, current_user_lat, current_user_lon, \
            trip_duration_days, estimated_budget, num_travelers, special_requests = get_user_inputs_ui()

            st.header("② 질문하기")
            user_query = st.text_input("어떤 여행을 계획하고 계신가요? (예: 가족과 함께 즐길 수 있는 자연 테마 여행)", value=st.session_state.current_input, key="user_input")

            if st.button("여행 계획 추천받기"):
                st.session_state.selected_conversation_index = None

                lat_to_invoke = current_user_lat
                lon_to_invoke = current_user_lon

                age_to_invoke = age
                travel_style_to_invoke = ', '.join(travel_style_list) if travel_style_list else '특정 없음'
                trip_duration_days_to_invoke = trip_duration_days
                estimated_budget_to_invoke = estimated_budget
                num_travelers_to_invoke = num_travelers
                special_requests_to_invoke = special_requests

                if lat_to_invoke is None or lon_to_invoke is None:
                    st.warning("위치 정보가 없으므로 답변을 생성할 수 없습니다. 위치 정보를 입력하거나 가져와 주세요.")
                elif not user_query.strip():
                    st.warning("질문을 입력해주세요.")
                else:
                    with st.spinner("최적의 여행 계획을 수립 중입니다..."):
                        try:
                            response = qa_chain.invoke({
                                "input": user_query,
                                "age": age_to_invoke,
                                "travel_style": travel_style_to_invoke,
                                "user_lat": lat_to_invoke,
                                "user_lon": lon_to_invoke,
                                "trip_duration_days": trip_duration_days_to_invoke,
                                "estimated_budget": estimated_budget_to_invoke,
                                "num_travelers": num_travelers_to_invoke,
                                "special_requests": special_requests_to_invoke
                            })

                            rag_result_text = response["answer"]

                            processed_output_lines = []
                            processed_place_names = set()
                            table_plan_text = ""
                            in_plan_section = False

                            for line in rag_result_text.split('\n'):
                                if "상세 여행 계획" in line and "일차 | 시간 | 활동" not in line:
                                    processed_output_lines.append(line)
                                    in_plan_section = True
                                    continue

                                if not in_plan_section:
                                    name_match = re.search(r"관광지 이름:\s*(.+)", line)
                                    if name_match:
                                        current_place_name = name_match.group(1).strip()
                                        if current_place_name not in processed_place_names:
                                            processed_output_lines.append(line)
                                            processed_place_names.add(current_place_name)

                                            found_place_data = tour_data_df[
                                                (tour_data_df['관광지명'].str.strip() == current_place_name) &
                                                (pd.notna(tour_data_df['위도'])) &
                                                (pd.notna(tour_data_df['경도']))
                                            ]
                                            if not found_place_data.empty:
                                                place_lat = found_place_data['위도'].iloc[0]
                                                place_lon = found_place_data['경도'].iloc[0]
                                                distance = haversine(lat_to_invoke, lon_to_invoke, place_lat, place_lon)
                                                processed_output_lines.append(f"- 사용자 위치 기준 거리(km): 약 {distance:.2f} km")
                                            else:
                                                processed_output_lines.append("- 사용자 위치 기준 거리(km): 정보 없음 (데이터 불일치 또는 좌표 누락)")
                                    else:
                                        if not re.search(r"거리\(km\):", line):
                                            processed_output_lines.append(line)
                                else:
                                    table_plan_text += line + "\n"

                            st.subheader("추천 결과 및 상세 여행 계획")
                            st.markdown("\n".join(processed_output_lines))

                            if table_plan_text.strip():
                                try:
                                    plan_lines = table_plan_text.strip().split('\n')
                                    
                                    if len(plan_lines) >= 2 and plan_lines[0].count('|') >= 2 and plan_lines[1].count('|') >= 2 and all(re.match(r'^-+$', s.strip()) for s in plan_lines[1].split('|') if s.strip()):
                                        header = [h.strip() for h in plan_lines[0].split('|') if h.strip()]
                                        data_rows = []
                                        for row_str in plan_lines[2:]:
                                            if row_str.strip() and row_str.startswith('|'):
                                                parsed_row = [d.strip() for d in row_str.split('|')]
                                                if parsed_row and parsed_row[0] == '':
                                                    parsed_row = parsed_row[1:]
                                                if parsed_row and parsed_row[-1] == '':
                                                    parsed_row = parsed_row[:-1]
                                                data_rows.append(parsed_row)

                                        if data_rows:
                                            if all(len(row) == len(header) for row in data_rows):
                                                temp_plan_df = pd.DataFrame(data_rows, columns=header)
                                                
                                                if '일차' in temp_plan_df.columns:
                                                    for i in range(1, len(temp_plan_df)):
                                                        if temp_plan_df.loc[i, '일차'] == temp_plan_df.loc[i-1, '일차']:
                                                            temp_plan_df.loc[i, '일차'] = ''
                                                
                                                    plan_df_styled = temp_plan_df.set_index('일차')
                                                    
                                                    st.subheader("🗓️추천여행계획표")
                                                    st.dataframe(plan_df_styled, use_container_width=True)
                                                else:
                                                    st.subheader("🗓️추천여행계획표표")
                                                    st.dataframe(temp_plan_df, use_container_width=True)
                                                    st.warning("여행 계획에 '일차' 컬럼이 없어 그룹화하여 표시할 수 없습니다.")
                                            else:
                                                st.warning("여행 계획 테이블의 행과 열의 수가 일치하지 않아 표를 생성할 수 없습니다. LLM 응답 형식을 확인해주세요.")
                                        else:
                                        #
                                            st.warning("여행 계획 테이블 내용을 파싱할 수 없습니다. LLM이 요청된 표 형식을 따르지 않았을 수 있습니다.")
                                    else:
                                        st.warning("여행 계획이 유효한 표 형식으로 제공되지 않았습니다.")
                                except Exception as parse_e:
                                    st.error(f"여행 계획 테이블 파싱 중 오류 발생: {parse_e}. LLM 응답 형식을 확인해주세요.")
                            else:
                                st.info("상세 여행 계획이 제공되지 않았습니다.")
                            
                            st.session_state.conversations.append({
                                "user_query": user_query,
                                "chatbot_response": rag_result_text,
                                "travel_style_selected": travel_style_to_invoke
                            })

                        except ValueError as ve:
                            st.error(f"체인 호출 중 오류 발생: {ve}. 입력 키를 확인해주세요.")
                        except Exception as e:
                            st.error(f"예상치 못한 오류 발생: {e}")

                st.session_state.current_input = "" # 입력창 초기화
