import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

# 데이터셋 로드
@st.cache_data()
def load_data():
    
    # Git LFS 명령 실행
    lfs_file_path = "wine_data_with_predictions_v5.csv"
    subprocess.run(["git", "lfs", "pull", lfs_file_path])
    df = pd.read_csv('wine_data_with_predictions_v5.csv')
    
    # 가격 정수로 변환
    df['price_int'] = df['price'].str.extract(r'(\d+)').astype('int')
    return df
# 데이터 불러오기
df = load_data()


#사용자 입력값 전처리
def preprocess_user_input(user_input):
    # 소문자로 변환
    user_input = user_input.lower()
    # 구두문자 제거
    user_input = user_input.translate(str.maketrans('', '', string.punctuation))
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(user_input)
    user_input = [word for word in word_tokens if word not in stop_words]
    # 공백 제거 및 문자열로 병합
    user_input = ' '.join(user_input)
    return user_input

# 컬럼 필터 함수
def filter_df_by_column(user_input, df, column):
    column_filtered_df = df.copy()
    user_input_split = user_input.split()
    column_values = df[column].str.lower().tolist()
    if any(value in user_input.lower() for value in column_values):
        column_filtered_df = df[df[column].str.lower().isin(user_input_split)]
    return column_filtered_df

# 유사한 와인 추천 함수
def recommend_wines(user_input, df, price_range=False):

    # 사용자 입력값 전처리
    user_input = preprocess_user_input(user_input)
    
    # price 검색
    price_filtered_df = df.copy()
    if price_range :
        # 선택한 범위에 해당하는 행 추출
        price_filtered_df = df[(df['price_int'] >= price_range) & (df['price_int'] <= price_range + 10)]

    # wine_type 검색
    wine_type_filtered_df=filter_df_by_column(user_input, df, "wine_type")
    # country 검색
    country_filtered_df=filter_df_by_column(user_input, df, "country")

    # price & wine_type 결과 합치기
    filtered_df = pd.merge(price_filtered_df, wine_type_filtered_df, how='inner')
    filtered_df = pd.merge(filtered_df, country_filtered_df, how='inner')
    
    # 유사도 계산을 위해 TF-IDF 벡터 생성
    # country, title, description, province, variety, winery
    all_values = filtered_df[['country', 'title', 'province', 'winery', 'description', 'variety']].values.flatten()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_values)
    # 사용자 입력값을 TF-IDF로 변환
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    # 유사도 계산
    similarity_scores = cosine_similarity(user_input_tfidf, tfidf_matrix[filtered_df.index]).flatten()
    # 유사도가 높은 순서대로 정렬하여 결과 반환
    filtered_df['similarity'] = similarity_scores
    filtered_df = filtered_df.sort_values(by='similarity', ascending=False)
    # 유사도 컬럼 제거
    filtered_df = filtered_df.drop(columns=['similarity'])
    
    return filtered_df

###############################################################################


# 대화형 UI 구성
st.title("Wine Recommender ChatBot")

# 이전 대화 기록을 저장할 세션 상태 초기화
if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []

# 사용자 입력값에 키워드가 포함되어 있는지 확인하는 함수
def has_dollar_sign(text):
    lowercase_text = text.lower()
    # 키워드 리스트
    price_keywords = ["price", "cost", "priced", "dollar", "dollars", "$"]
    # 주어진 키워드가 입력값에 포함되어 있는지 확인
    contains_keyword = any(keyword in lowercase_text for keyword in price_keywords)
    return contains_keyword

# 사용자 입력값
user_input = st.text_input("Enter a search term and press Enter")

# 이전 대화 로그에서 가장 최근의 챗봇 응답 가져오기
previous_bot_response = None
for entry in st.session_state.conversation_log:
    if "bot" in entry:
        previous_bot_response = entry["bot"]
        break
# 이전 검색 결과가 있는 경우
if previous_bot_response is not None:
    # 이전 검색 결과를 현재의 검색 데이터셋으로 사용
    df = previous_bot_response
else:
    # 이전 검색 결과가 없는 경우는 전체 데이터셋을 사용
    df = df
    
# $가 포함되어 있으면 가격대 선택을 위한 셀렉트박스를 보여줌
if has_dollar_sign(user_input) :
    price_range = st.sidebar.selectbox("Select a range of values", range(0, 3301, 10))
    if price_range :
        # 새로운 검색 수행
        recommend_wines = recommend_wines(user_input, df, price_range)
        bot_response = recommend_wines.reset_index(drop=True)
        bot_response.index = range(1, len(bot_response) + 1)
    
        # 로그에 추가
        st.session_state.conversation_log.insert(0, {"user": user_input ,"price":f'{price_range}$ ~ {price_range+10}$', "bot": bot_response})
        
        # 대화 로그 출력
        for entry in st.session_state.conversation_log:
            # 사용자 입력과 챗봇 응답이 쌍을 이루도록 출력
            if "user" in entry and "bot" in entry:
                # 사용자 입력
                st.write(f'<div style="display:flex; flex-direction:column; align-items:flex-end; margin:20px 0 10px"><div style="background-color: #f0f0f0; border-radius:20px; padding:5px 15px">{entry["user"]}<p style="margin:0">Selected price range : {entry["price"]}</p></div></div>', unsafe_allow_html=True)   
                # 챗봇 응답
                st.write(entry["bot"][['title', 'country', 'wine_type', 'description', 'price']].head(10))
else : 
    if user_input :
        # 새로운 검색 수행
        recommend_wines = recommend_wines(user_input, df)
        bot_response = recommend_wines.reset_index(drop=True)
        bot_response.index = range(1, len(bot_response) + 1)
        
        # 로그에 추가
        st.session_state.conversation_log.insert(0, {"user": user_input,"price":None, "bot": bot_response})
        
        # 대화 로그 출력
        for entry in st.session_state.conversation_log:
            # 사용자 입력과 챗봇 응답이 쌍을 이루도록 출력
            if "user" in entry and "bot" in entry:
                # 사용자 입력
                st.write(f'<div style="display:flex; flex-direction:column; align-items:flex-end; margin:20px 0 10px"><div style="background-color: #f0f0f0; border-radius:20px; padding:5px 15px">{entry["user"]}</div></div>', unsafe_allow_html=True)   
                # 챗봇 응답
                st.write(entry["bot"][['title', 'country', 'wine_type', 'description', 'price']].head(10))




    
    
    
    
    

