# 🧭 위치기반 관광지 추천 챗봇

> ✨ 지금 당장 떠나고 싶은 당신을 위한 여행 메이트  
---

## 📌 소개

**“어디로 갈까?” 고민하는 여행자들을 위한 위치기반 관광지 추천 챗봇입니다.**  
특히 MBTI 유형 중 **P 타입**처럼 즉흥적이고 자유로운 여행을 즐기는 분들을 위한 서비스입니다.

---

## 👤 대상 사용자

- MBTI **P 유형** 사용자
- **즉흥적으로 여행을 가고 싶은 사람**
- **어디를 갈지 정하지 못한 사람**
- **내 주변 관광지를 추천받고 싶은 사람**

---

## 🔍 주요 기능

### 📍 실시간 위치 기반 관광지 추천
- `streamlit_geolocation`을 통해 **사용자의 현재 위치**를 실시간으로 받아옵니다.
- 위치를 기반으로 **주변 관광지**를 추천합니다.

### 🧭 여행 성향 & 나이 기반 맞춤 추천
- **여행 성향**과 **나이대**를 기반으로
- 사용자에게 맞는 관광지를 **개인화된 방식**으로 추천합니다.

### 🗓️ 여행 일정 및 예산 설계
- **원하는 여행 일수**와 **예산**을 입력하면
- **추천 루트 및 여행 계획표**를 자동으로 생성합니다.

### 💬 이전 대화 저장
- **이전 대화 내역**을 저장하여
- 다시 불러오거나 이어서 사용할 수 있습니다.

### 📄 여행 계획표 다운로드
- **여행 계획표를 파일로 다운로드**하여 저장할 수 있습니다.
- 언제 어디서든 오프라인으로 확인 가능!

---

## 🚀 체험해보기

[👉직접 체험해보세요](https://aiserviceteam-lucky.streamlit.app/ )



> 이 챗봇은 사용자 중심의 여행 설계를 도와주는 도구로,  
> 자유로운 여행을 꿈꾸는 당신을 위해 만들어졌습니다.
🚀 시스템 개요: 데이터 처리 및 검색 흐름

저희 시스템은 다양한 형태의 **관광지 데이터**를 통합하고, 이를 효율적으로 **검색하여 사용자의 질의에 정확하게 응답**하는 것을 목표로 합니다. 이 과정은 크게 **데이터 로드 및 전처리, 그리고 데이터 검색 및 벡터화**단계로 나뉩니다.

🛠️ 데이터 로드 및 전처리 함수
-
load_specific_tour_data 함수는 여러 CSV 파일에 분산된 관광지 정보를 통합하고, 필요한 형태로 정제하는 역할을 합니다. @st.cache_data 데코레이터를 사용하여 Streamlit 앱의 성능을 최적화합니다.

```python
@st.cache_data
def load_specific_tour_data(file_paths_list):
    # 지정된 CSV 파일 목록을 로드하고, 모든 파일에 CP949 인코딩을 적용하여 병합합니다.
    combined_df = pd.DataFrame()
    if not file_paths_list:
        st.error("로드할 관광지 CSV 파일 경로가 지정되지 않았습니다. TOUR_CSV_FILES를 확인해주세요.")
        st.stop()

    for file_path in file_paths_list:
        if not os.path.exists(file_path):
            st.warning(f"'{file_path}' 파일을 찾을 수 없어 건너뜁니다. (Streamlit Cloud에서는 해당 파일들이 Git 리포지토리에 포함되어야 합니다.)")
            continue

        current_encoding = 'cp949'  # 인코딩 수정

        try:
            df = pd.read_csv(file_path, encoding=current_encoding)
            df.columns = df.columns.str.strip()

            if "위도" not in df.columns or "경도" not in df.columns:
                st.warning(f"'{os.path.basename(file_path)}' 파일은 '위도', '경도' 컬럼이 없어 건너뜁니다.")
                continue

            name_col = None
            for candidate in ["관광지명", "관광정보명", "관광지"]:
                if candidate in df.columns:
                    name_col = candidate
                    break

            if name_col is None:
                df["관광지명"] = "이름 없음"
            else:
                df["관광지명"] = df[name_col]

            address_col = None
            for candidate in ["정제도로명주소", "정제지번주소", "소재지도로명주소", "소재지지번주소", "관광지소재지지번주소", "관광지소재지도로명주소"]:
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
```

🎯 주요 기능
-
- **CSV 파일 통합:**
file_paths_list에 명시된 여러 CSV 파일을 CP949 인코딩으로 로드하여 하나의 데이터프레임으로 병합합니다.
- **데이터 유효성 검사**: 각 파일에 필수 컬럼인 위도와 경도가 있는지 확인합니다.
- **컬럼 표준화**: 관광지명과 소재지도로명주소와 같이 의미는 같지만 이름이 다른 컬럼들을 표준화하여 데이터 일관성을 확보합니다.
- **오류 처리**: 파일이 없거나 로드 중 오류가 발생하면 경고 메시지를 표시하고 해당 파일을 건너뜁니다.
- **캐싱**: @st.cache_data 데코레이터를 사용하여 함수 실행 결과를 캐싱하므로, 데이터가 변경되지 않는 한 불필요한 재로딩을 방지하여 앱의 로딩 속도를 향상시킵니다

🔍 데이터 검색 및 벡터화 (Retrieval)
-
이 섹션에서는 RAG(Retrieval Augmented Generation) 아키텍처를 활용하여 다양한 CSV 파일에 분산된 관광지 정보를 통합하고 효율적으로 검색합니다. 이는 LLM(Large Language Model)이 단순히 학습된 지식에 의존하는 것이 아니라, 실제 데이터를 기반으로 답변을 생성하게 하는 핵심 단계입니다.

**1. 다양한 CSV 파일 통합 및 벡터화**
경기도역사관광지현황.csv, 경기도자연관광지현황.csv 등 여러 CSV 파일에 흩어진 방대한 관광지 정보를 효과적으로 활용하기 위해 다음 과정을 거칩니다.

- CSVLoader: LangChain의 CSVLoader를 사용하여 각 CSV 파일의 데이터를 로드합니다. 이때, cp949 인코딩을 적용하여 한글 데이터 처리의 안정성을 높입니다.
- RecursiveCharacterTextSplitter: 로드된 문서는 RecursiveCharacterTextSplitter를 이용해 일정한 크기(chunk_size=250, chunk_overlap=50)로 분할됩니다. 이는 문서의 내용을 의미 단위로 쪼개어 벡터화에 적합한 형태로 만듭니다.
- 데이터 임베딩: 이 과정을 통해 관광지 데이터가 벡터 공간에 임베딩되어 저장됩니다.
**2. FAISS를 활용한 효율적인 정보 검색**
FAISS는 대규모 벡터 데이터를 빠르게 검색할 수 있게 해주는 라이브러리입니다. 사용자의 질의가 들어오면 다음과 같이 작동합니다.

질의 벡터 변환: 사용자의 질의 또한 OpenAI 임베딩을 통해 벡터로 변환됩니다.
의미적 관련성 검색: 변환된 질의 벡터를 사용하여 FAISS 벡터 스토어에서 의미적으로 가장 관련성이 높은 관광지 정보(context)를 빠르게 찾아냅니다.
      

📦 벡터스토어 로딩 및 캐싱
-
아래 코드는 관광지 정보를 벡터화하여 저장하고, 필요할 때 빠르게 로드하거나 새로 생성하는 함수입니다.
load_and_create_vectorstore_from_specific_files 함수
이 함수는 지정된 CSV 파일 목록을 사용하여 벡터스토어를 생성합니다.
```python
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


get_vectorstore_cached 함수
-
이 함수는 이미 생성된 벡터 DB가 있다면 이를 로드하여 불필요한 재구축을 막아 성능을 최적화합니다.

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
```
🎯 주요 기능
-
- **캐싱 및 재활용**: @st.cache_resource 데코레이터를 사용하여 벡터스토어 생성 작업을 캐싱합니다. get_vectorstore_cached 함수는 VECTOR_DB_PATH에 벡터스토어가 이미 존재하면 이를 로드하여 불필요한 재계산을 방지합니다.
- **FAISS 로컬 저장 및 로드:**
생성된 벡터스토어는 FAISS.save_local()을 통해 지정된 경로에 저장되며, FAISS.load_local()을 통해 다시 로드될 수 있습니다. allow_dangerous_deserialization=True는 최신 FAISS 버전에서 필요한 설정입니다.
- **오류 복구:**
기존 벡터스토어 로딩에 실패할 경우, 자동으로 load_and_create_vectorstore_from_specific_files 함수를 호출하여 새로 생성함으로써 시스템의 안정성을 높입니다.
- **텍스트 분할 및 임베딩:**
  RecursiveCharacterTextSplitter를 사용하여 문서를 청크로 분할하고, OpenAIEmbeddings()를 사용하여 벡터로 변환하는 과정은 load_and_create_vectorstore_from_specific_files 함수 내에서 이루어집니다.


📐 거리 계산 및 사용자 입력 처리
-
##  Haversine 거리 계산 함수
haversine(lat1, lon1, lat2, lon2) 함수는 두 지점 간의 거리를 구하는 데 사용되는 **하버사인 공식(Haversine formula)**을 구현한 것입니다. 이는 지구 표면의 두 점 사이의 최단 거리를 구하는 데 유용하며,

## Haversine distance function 
```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance
```

# 🎯 주요 기능

- **지구 반지름:**
  지구의 평균 반지름 6371km를 사용하여 거리를 계산합니다.
- **라디안 변환:**
  위도와 경도 값을 삼각 함수 계산을 위해 라디안 단위로 변환합니다.
- **거리 계산:**
  하버사인 공식을 적용하여 두 지점 간의 **대원 거리(Great-circle distance)**를 정확하게 산출합니다.

사용자 입력 및 UI 로직 함수
-
get_user_inputs_ui() 함수는 Streamlit을 활용하여 사용자로부터 다양한 여행 관련 정보를 입력받는 UI를 제공합니다. 이 함수를 통해 수집된 정보는 LLM이 사용자 맞춤형 여행 계획을 수립하는 데 중요한 기반 데이터로 활용됩니다
```python
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

```

💡 LLM 프롬프트 구성 및 검색 전략
-
이 코드에서 수집된 **사용자의 질의와 검색된 관련 문서 (context)**, 그리고 **get_user_inputs_ui() 함수를 통해 얻은 다양한 사용자 입력(나이, 여행 스타일, 현재 위치 등)**은 gpt-4o LLM의 **동적인 프롬프트**를 구성하는 핵심 요소입니다.

이를 통해 LLM은 단순히 일반적인 지식에 기반한 답변을 넘어, **사용자의 특성과 검색된 실제 관광지 데이터를 바탕으로 맞춤형 여행 계획을 수립하도록 지시**받게 됩니다.

search_kwargs={"k": 15}: 벡터 스토어에서 검색 시 **가장 유사한 15개의 문서를 가져오도록 설정**합니다. 이는 LLM이 더 풍부하고 다양한 정보를 기반으로 답변을 생성할 수 있도록 하여, 답변의 질과 관련성을 크게 향상시킵니다.
이러한 접근 방식은 RAG 아키텍처의 강점을 최대한 활용하여, 개인화되고 정확한 여행 추천 시스템을 구축하는 데 기여합니다.


🧠 추천 로직 및 LLM 프롬프트 구성
-
이 섹션은 시스템의 핵심적인 추천 로직을 담당하며, LangChain의 create_retrieval_chain을 활용하여 사용자 질의에 대한 맞춤형 답변을 생성합니다. @st.cache_resource **데코레이터**를 통해 LLM 체인 객체를 효율적으로 캐싱하여 성능을 최적화합니다.

```python
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
"""
```

🎯 주요 기능 및 특징
-
- # LLM (Large Language Model) 설정:
 ChatOpenAI(model_name="gpt-4o", temperature=0.7)를 사용하여 OpenAI의 최신 모델인 gpt-4o를 활용합니다. temperature=0.7은 답변의 창의성을 적절하게 조절하여 너무 정형화되지 않으면서도 관련성 높은 결과를 유도합니다.
 
- # 동적 프롬프트 구성:
- 사용자 질의: {input} 변수를 통해 사용자의 직접적인 질문이 포함됩니다.
검색된 문서 (Context): {context} 변수에 벡터 스토어에서 검색된 관련 관광지 데이터가 삽입됩니다. 이는 LLM이 실제 데이터를 기반으로 답변을 생성하게 하는 핵심 요소입니다.

- 다양한 사용자 입력: age, travel_style, user_lat, user_lon, trip_duration_days, estimated_budget, num_travelers, special_requests 등 사용자로부터 입력받은 상세 정보들이 프롬프트에 포함됩니다.
  
- 맞춤형 여행 계획 지시: 프롬프트는 LLM에게 다음과 같은 구체적인 지침을 제공하여 개인화된 답변을 유도합니다.
  
- 위치 기반 우선 추천: 사용자 현재 위치에서 가까운 장소를 최우선으로 고려하여 추천하도록 명시합니다. (user_lat, user_lon 활용)
  
- 거리 언급 제외: 시스템이 거리를 자동으로 계산하여 별도로 제공하므로, LLM의 답변에서는 거리를 직접 언급하지 않도록 지시합니다.
추천 관광지 정보 명시: 추천 관광지의 이름, 주소, 주요 시설/특징을 포함하도록 요구합니다.
상세 여행 계획: 여행 기간을 고려하여 일자별 상세 계획을 표 형식으로 구성하도록 지시하며, 각 날짜별 활동, 예상 장소, 이동 방법 등을 포함하도록 합니다.

- 예산 고려: 예상 예산을 바탕으로 적절한 식사 장소나 활동을 제안하도록 유도합니다.
테이블 형식 강제: 답변의 가독성을 높이기 위해 일차, 시간, 활동, 예상 장소, 이동 방법 컬럼을 가진 표 형식으로 출력하도록 명확히 지시합니다. 또한, '일차' 셀 병합을 위한 특정 형식까지 안내하여 Streamlit에서의 시각적 완성도를 높입니다.

**create_stuff_documents_chain**: 검색된 모든 문서를 하나의 문자열로 묶어 LLM에 전달하는 역할을 합니다.
_vectorstore.as_retriever(search_kwargs={"k": 15}): 벡터 스토어에서 사용자 질의와 가장 유사한 15개의 문서를 검색하도록 설정합니다. 이는 LLM이 더 풍부하고 다양한 정보를 기반으로 답변을 생성할 수 있도록 하여, 답변의 질과 관련성을 크게 향상시킵니다.

**create_retrieval_chain**: retriever와 document_chain을 연결하여, 정보 검색(Retrieval)과 응답 생성(Generation)을 통합하는 RAG 파이프라인을 구축합니다.
  
이러한 정교한 프롬프트 구성과 체인 설정은 단순한 정보 제공을 넘어, 사용자의 개별적인 요구사항과 현재 상황을 반영한 실제적인 여행 계획을 제안하는 데 기여합니다.
message.txt
25KB
