import numpy as np
import pandas as pd
import json
import asyncio
import discord
from discord.ext import commands

import os
from dotenv import load_dotenv

load_dotenv()
#사용자 기반 협업 필터링
#아이템 기반은 나중에 만들고 추후에 하이브리드 필터링 연구까지 하고 제작하기.

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

data = pd.merge(ratings, meta, on='movieId', how='inner') #pd.merge(데이터프레임1, 데이터프레임2,on=대상,how=어떤식)
#pd.merge() 두 개의 데이터 프레임 병합, movieId를 기준으로 inner 한 줄로 저장
#data.head()

matrix = data.pivot_table(index='userId', columns='original_title', values='rating')
#피벗 테이블 제작 칼럼 - 가로 , 인덱스 - 세로
#matrix.head(20)

GENRE_WEIGHT = 0.1  #피어슨 상관계수

def pearsonR(s1, s2):
    s1_c = s1 - s1.mean() #사용자1 - 사용자 1 행의 전체 평균값을 뺌
    s2_c = s2 - s2.mean() #사용자2 - 사용자 2 행의 전체 평균값을 뺌
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2)) #사용자 1과 2의 유사도를 판단하기 위해서 
    # s1_c와 s2_c를 곱해주고 이를 모두 더해줌, 그리고 이를 s1_c의 제곱을 더한값 * s2_c의 제곱을 더한 값에 제곱근을 씌워 나눠주고


def recommend(input_movie, matrix, n, similar_genre=True): #영화 이름, 행렬 ,몇 개의 영화를 찾는지 , 장르에 따라 가중치를 줄 건지
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

token = os.getenv('TOKEN')


client = commands.Bot(command_prefix='/') #명령어 인식

@client.event
async def on_ready():
    print(client.user.name)
    print('봇 시작됨')
    game = discord.Game("/도움말")
    await client.change_presence(status=discord.Status.online, activity=game)


@client.command(name='도움말')
async def on_message(message):
    embed = discord.Embed(colour=discord.Colour.blue(), title = "《 인공지능 영화 추천 봇 》", description=" 본 시스템은 협업 필터링을 이용한 딥러닝 추천시스템으로 구성되었습니다. ")
    embed.add_field(name='/영화추천 [좋아하는 영화 이름]',value='자신이 입력한 영화 데이터를 기반으로 유사한 영화를 추천합니다. \n 단, 띄어쓰기는 _ 언더바로 표시해주세요!',inline=False)

    await message.channel.send(embed=embed)

@client.command(name='영화추천')
async def on_message(message, Inputname):
    #언더바 전처리
    moviename = Inputname.replace("_"," ")
    await message.channel.send(f"추천할 영화를 검색 중입니다.")
    recommend_result = recommend(str(moviename), matrix, 10, similar_genre=True)
    pd.DataFrame(recommend_result, columns = ['Title', 'Correlation', 'Genre'])

    p_title = {}
    p_percent = {}
    p_genres = {}

    for i in range (0,5):
        p_title[i] = recommend_result[i][0]
        p_percent[i] = int(float(recommend_result[i][1])*100)
        p_genres[i] = " / ".join(recommend_result[i][2])
        
    for j in range (0,3):
        embed = discord.Embed(colour=discord.Colour.blue(), title = "《 영화 추천 결과 》")
        embed.add_field(name='영화 이름',value=f'{p_title[j]}',inline=False)
        embed.add_field(name='유사도',value=f'{p_percent[j]}%',inline=False)
        embed.add_field(name='장르',value=f'{p_genres[j]}',inline=False)
        await message.channel.send(embed=embed)

@on_message.error
async def roll_error(ctx, error):
    await ctx.send("띄어쓰기를 _로 표시했는지 확인해주세요.입력하신 영화는 현재 보유 데이터 셋에 존재하지 않거나, 잘못 입력되었습니다.")

client.run(token)