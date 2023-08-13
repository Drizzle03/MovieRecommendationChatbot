import numpy as np
import pandas as pd
import json

# 데이터 프레임에서 필요한 데이터 추출 및 가공하고, 영화 평가 값 저장 ======================================
meta = pd.read_csv('.data/movies_metadata.csv',low_memory = False,dtype={'CODE':np.str}) 
#영화에 대한 메타 데이터

meta = meta[['id', 'original_title', 'original_language', 'genres']] #메타 데이터 추출 및 저장 id / 제목 / 언어 / 장르
meta = meta.rename(columns={'id':'movieId'}) # id 데이터를 가독성 높게 movieId로 리네임
meta = meta[meta['original_language'] == 'en'] # 영어 en 영화만 불러옴

ratings = pd.read_csv('.data/ratings_small.csv') 
#실제 사용이 아니기에 평가 데이터가 작은 것을 씀
ratings = ratings[['userId','movieId','rating']] #유저 고유번호 / 영화 고유번호 / 평가

ratings.describe() #describe : 간략한 정보 출력 (총 갯수 / 평균값 / 표준편차 / 최솟값 등)

meta.movieId = pd.to_numeric(meta.movieId, errors='coerce') #문자열을 숫자 타입으로 변환 string => int
ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce') #문자열을 숫자 타입으로 변환 string => int

#========================================================================================================

def parse_genres(genres_str): #장르 리스트를 정돈해줌
    genres = json.loads(genres_str.replace('\'','"')) # 작은 따옴표 => 큰 따옴표 변경 
    # 작은 따옴표, 큰 따옴표 모두 문자열을 표시한다는 것에서 차이가 없지만 작은 따옴표의 경우 하나의 단위를 표시하는 문자열,
    # 큰 따옴표의 경우 문법적으로 의미를 두지 않는 문자열을 표현할 때 사용

    genres_list = [] 

    for g in genres:
        genres_list.append(g['name']) #genres_list 리스트에 장르 추가 genres_list에 name값 추가, (위치['리스트값'])
        # [ Animation, Comedy, Family ]와 같이 출력


        # array = '{"마실꺼": ["커피", "차", "물"]}'
        #data = json.loads(array)
 
        #for element in data['마실꺼']:
        #print (element) # 커피, 차, 물
    return genres_list

meta['genres'] = meta['genres'].apply(parse_genres) #장르 리스트 부분 각 행에 해당 함수 적용

data = pd.merge(ratings, meta, on='movieId', how='inner') 
#pd.merge() 두 개의 데이터 프레임 병합, movieId를 기준으로 inner 한 줄로 저장
#data.head()

matrix = data.pivot_table(index='userId', columns='original_title', values='rating')
#피벗 테이블 제작 칼럼 - 가로 , 인덱스 - 세로
#matrix.head(20)

GENRE_WEIGHT = 0.1

def pearsonR(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))

def recommend(input_movie, matrix, n, similar_genre=True): #영화 이름, 행렬 , 행 크기 , 유사한 것을 찾는지
    input_genres = meta[meta['original_title'] == input_movie]['genres'].iloc(0)[0] #데이터셋의 영화 이름과 인풋값이 같을 때 

    result = []
    for title in matrix.columns:
        if title == input_movie:
            continue

        # rating comparison
        cor = pearsonR(matrix[input_movie], matrix[title])
        
        # genre comparison
        if similar_genre and len(input_genres) > 0:
            temp_genres = meta[meta['original_title'] == title]['genres'].iloc(0)[0]

            same_count = np.sum(np.isin(input_genres, temp_genres))
            cor += (GENRE_WEIGHT * same_count)
        
        if np.isnan(cor):
            continue
        else:
            result.append((title, '{:.2f}'.format(cor), temp_genres))
            
    result.sort(key=lambda r: r[1], reverse=True)

    return result[:n]


inputname = 'Chances_Are'
moviename = inputname.replace("_"," ")
recommend_result = recommend(str(moviename), matrix, 10, similar_genre=True)

pd.DataFrame(recommend_result, columns = ['Title', 'Correlation', 'Genre'])

p_title = {}
p_percent = {}
p_genres = {}

for i in range (0,5):
    p_title[i] = recommend_result[i][0]
    p_percent[i] = int(float(recommend_result[i][1])*100)
    p_genres[i] = recommend_result[i][2]
    
for j in range (0,5): 
    print(f"영화 이름 : {p_title[j]}\n유사도 : {(p_percent[j])}%\n장르 : {p_genres[j]}")