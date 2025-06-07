import streamlit as st

from streamlit_geolocation import streamlit_geolocation

import pandas as pd

from math import radians, sin, cos, sqrt, atan2

import os

import re

import glob



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

load_dotenv()



st.set_page_config(page_title="관광지 추천 챗봇", layout="wide")



# --- 파일 경로 정의 (상수) ---

VECTOR_DB_PATH = "faiss_tourist_attractions"



# 로드할 개별 관광지 CSV 파일 목록을 직접 지정합니다.

# **여기를 실제 CSV 파일 경로에 맞게 수정해주세요!**

# Streamlit Cloud에서는 상대 경로를 사용해야 합니다.

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

# 모든 필수 데이터 파일이 존재하는지 확인합니다.

required_files = TOUR_CSV_FILES

for f_path in required_files:

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

        st.success("✅ OpenAI API 키를 Streamlit Secrets에서 성공적으로 로드했습니다.")

        return st.secrets['OPENAI_API_KEY']

    else:

        load_dotenv() # 로컬 개발 시 .env 파일에서 로드 시도

        api_key = os.getenv("OPENAI_API_KEY")

        if api_key:

            st.success("✅ OpenAI API 키를 환경 변수(.env 파일 또는 시스템 환경 변수)에서 성공적으로 로드했습니다.")

        else:

            st.error("❌ OpenAI API 키를 찾을 수 없습니다. Streamlit Cloud에서는 secrets.toml에 키를 설정하거나, 로컬에서는 .env 파일을 확인해주세요.")

        return api_key





def initialize_streamlit_app():

    """Streamlit 앱의 기본 페이지 설정 및 제목을 초기화합니다."""

    st.title("🗺️ 위치 기반 관광지 추천 및 여행 계획 챗봇")



# --- 2. 데이터 로드 및 전처리 함수 ---

@st.cache_data

def load_specific_tour_data(file_paths_list): # utf8_files 파라미터 제거

    """지정된 CSV 파일 목록을 로드하고, 모든 파일에 CP949 인코딩을 적용하여 병합합니다."""

    combined_df = pd.DataFrame()



    if not file_paths_list:

        st.error("로드할 관광지 CSV 파일 경로가 지정되지 않았습니다. `TOUR_CSV_FILES`를 확인해주세요.")

        st.stop()



    for file_path in file_paths_list:

        if not os.path.exists(file_path):

            st.warning(f"'{file_path}' 파일을 찾을 수 없어 건너뜱니다. (Streamlit Cloud에서는 해당 파일들이 Git 리포지토리에 포함되어야 합니다.)")

            continue



        # 모든 파일에 CP949 인코딩 적용

        current_encoding = 'cp949'



        try:

            df = pd.read_csv(file_path, encoding=current_encoding)

            df.columns = df.columns.str.strip()



            if "위도" not in df.columns or "경도" not in df.columns:

                st.warning(f"'{os.path.basename(file_path)}' 파일은 '위도', '경도' 컬럼이 없어 건너뜁니다.")

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

def load_and_create_vectorstore_from_specific_files(tour_csv_files_list): # utf8_files 파라미터 제거

    """지정된 CSV 파일 목록을 사용하여 벡터스토어를 생성합니다."""

    all_city_tour_docs = []

    for file_path in tour_csv_files_list:

        if not os.path.exists(file_path):

            st.warning(f"벡터스토어 생성을 위해 '{file_path}' 파일을 찾을 수 없어 건너뜁니다.")

            continue



        # 모든 파일에 CP949 인코딩 적용

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

def get_vectorstore_cached(tour_csv_files_list): # utf8_files 파라미터 제거

    """캐시된 벡터스토어를 로드하거나 새로 생성합니다."""

    cache_key = tuple(sorted(tour_csv_files_list)) # 캐시 키에서 utf8_files 제거



    if os.path.exists(VECTOR_DB_PATH):

        try:

            # st.info("기존 벡터 DB를 로드 중...") # Removed this line

            return FAISS.load_local(

                VECTOR_DB_PATH,

                OpenAIEmbeddings(),

                allow_dangerous_deserialization=True

            )

        except Exception as e:

            st.warning(f"기존 벡터 DB 로딩 실패: {e}. 새로 생성합니다.")

            return load_and_create_vectorstore_from_specific_files(tour_csv_files_list) # 인자 제거

    else:

        # st.info("새로운 벡터 DB를 생성 중...") # Removed this line

        return load_and_create_vectorstore_from_specific_files(tour_csv_files_list) # 인자 제거





# --- Haversine distance function ---

def haversine(lat1, lon1, lat2, lon2):

    R = 6371  # Radius of Earth in kilometers

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

        age = st.selectbox("나이대 선택", ["10대", "20대", "30대", "40대", "50대 이상"], key='age_selectbox')

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



    # 1. 문서 체인 (프롬프트 + 문서 결합) 생성

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

1.  **관광지 추천:** 질문에 부합하고, 사용자 위치에서 가까운 1~3개의 주요 관광지를 추천하고, 각 관광지에 대한 다음 정보를 제공하세요.

    * 관광지 이름: [관광지명]

    * 주소: [주소]

    * 주요 시설/특징: [정보]

    **[참고: 사용자 위치 기준 거리는 시스템이 자동으로 계산하여 추가할 것이므로, 이 항목은 제외합니다.]**

      

2.  **추천된 관광지를 포함하여, 사용자 정보와 질문에 기반한 {trip_duration_days}일간의 상세 여행 계획을 일자별로 구성해 주세요.**

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



**1일차:**

- 오전 (9:00 - 12:00): [관광지명 1] 방문 (예: 역사 탐방, 박물관 관람).

- 점심 (12:00 - 13:00): [근처 식당명] (예: 한식 맛집, {estimated_budget}원에 적합한 메뉴)

- 오후 (13:00 - 17:00): [관광지명 2] 방문 (예: 자연 경관 감상, 산책). [관광지명 1]에서 버스 30분 이동.

- 저녁 (17:00 이후): [특정 활동 또는 자유 시간]



**2일차:**

- ... (이후 날짜별 계획) ...

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



    initialize_streamlit_app()



    vectorstore = get_vectorstore_cached(TOUR_CSV_FILES) # 인자 제거

    qa_chain = get_qa_chain(vectorstore)



    tour_data_df = load_specific_tour_data(TOUR_CSV_FILES) # 인자 제거



    if "messages" not in st.session_state:

        st.session_state.messages = []

    if "current_input" not in st.session_state:

        st.session_state.current_input = ""

    if "selected_message_index" not in st.session_state:

        st.session_state.selected_message_index = None



    # Sidebar for previous conversations

    with st.sidebar:

        st.subheader("💡 이전 대화")

        if st.session_state.messages:

            for i, m in enumerate(reversed(st.session_state.messages)):

                display_index = len(st.session_state.messages) - 1 - i

                role = "🙋‍♂️ 사용자" if m["role"] == "user" else "🤖 챗봇"

                

                # Create a clickable button for each message

                if st.button(f"{role} (대화 {display_index + 1})", key=f"sidebar_msg_{i}"):

                    st.session_state.selected_message_index = display_index

                    st.experimental_rerun() # Rerun to display the selected message



        else:

            st.info("이전 대화가 없습니다.")



    # Main content area

    age, travel_style_list, current_user_lat, current_user_lon, \

    trip_duration_days, estimated_budget, num_travelers, special_requests = get_user_inputs_ui()



    st.header("② 질문하기")

    user_query = st.text_input("어떤 여행을 계획하고 계신가요? (예: 가족과 함께 즐길 수 있는 자연 테마 여행)", value=st.session_state.current_input, key="user_input")



    if st.button("여행 계획 추천받기"):

        st.session_state.messages.append({"role": "user", "content": user_query})



        lat_to_invoke = current_user_lat

        lon_to_invoke = current_user_lon



        age_to_invoke = age

        travel_style_to_invoke = ', '.join(travel_style_list) if travel_style_list else '특정 없음'

        trip_duration_days_to_invoke = trip_duration_days

        estimated_budget_to_invoke = estimated_budget

        num_travelers_to_invoke = num_travelers

        special_requests_to_invoke = special_requests



        st.write(f"**DEBUG: Invoke Parameters**")

        st.write(f"input (query): {user_query}")

        st.write(f"age: {age_to_invoke}")

        st.write(f"travel_style: {travel_style_to_invoke}")

        st.write(f"user_lat: {lat_to_invoke}")

        st.write(f"user_lon: {lon_to_invoke}")

        st.write(f"trip_duration_days: {trip_duration_days_to_invoke}")

        st.write(f"estimated_budget: {estimated_budget_to_invoke}")

        st.write(f"num_travelers: {num_travelers_to_invoke}")

        st.write(f"special_requests: {special_requests_to_invoke}")





        if lat_to_invoke is None or lon_to_invoke is None:

            st.warning("위치 정보가 없으므로 답변을 생성할 수 없습니다. 위치 정보를 입력하거나 가져와 주세요.")

            st.session_state.messages.append({"role": "assistant", "content": "위치 정보가 없으므로 답변을 생성할 수 없습니다."})

        elif not user_query.strip():

            st.warning("질문을 입력해주세요.")

            st.session_state.messages.append({"role": "assistant", "content": "질문을 입력해주세요."})

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



                    for line in rag_result_text.split('\n'):

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

                                pass

                        else:

                            if not re.search(r"거리\(km\):", line):

                                processed_output_lines.append(line)



                    final_display_text = "\n".join(processed_output_lines)



                    st.session_state.messages.append({"role": "assistant", "content": final_display_text})

                    st.subheader("✅ 추천 결과 및 상세 여행 계획")

                    st.markdown(final_display_text)



                except ValueError as ve:

                    st.error(f"체인 호출 중 오류 발생: {ve}. 입력 키를 확인해주세요.")

                    st.session_state.messages.append({"role": "assistant", "content": f"오류가 발생했습니다: {ve}"})

                except Exception as e:

                    st.error(f"예상치 못한 오류 발생: {e}")

                    st.session_state.messages.append({"role": "assistant", "content": f"예상치 못한 오류가 발생했습니다: {e}"})



        st.session_state.current_input = ""



    # Display selected previous conversation

    if st.session_state.selected_message_index is not None:

        st.subheader("선택된 대화 내용")

        selected_message = st.session_state.messages[st.session_state.selected_message_index]

        role = "🙋‍♂️ 사용자" if selected_message["role"] == "user" else "🤖 챗봇"

        st.markdown(f"**{role}:**")

        st.markdown(selected_message['content'])

        st.markdown("---")
