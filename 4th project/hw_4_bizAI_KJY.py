# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split    
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import os, csv


def fn_data_split(data_df, train_file_name, vali_file_name, test_file_name):  
    
    major_df = data_df[data_df['label'] == 1].sample(frac = 0.14, random_state=1)
    minor_df = data_df[data_df['label'] == 0]   
  
    major_train, major_test = train_test_split(major_df, test_size = 0.4, random_state = 42)
    major_test, major_valid = train_test_split(major_test, test_size = 0.5, random_state = 42)
    minor_train, minor_test = train_test_split(minor_df, test_size = 0.4, random_state = 42)
    minor_test, minor_valid = train_test_split(minor_test, test_size = 0.5, random_state = 42)
    
    fin_train = pd.concat([major_train, minor_train])
    fin_test = pd.concat([major_test, minor_test])
    fin_valid = pd.concat([major_valid, minor_valid])
    

    fin_train.to_csv(train_file_name, header=True, index = False, encoding='cp949')
    fin_test.to_csv(test_file_name, header=True, index = False, encoding='cp949')
    fin_valid.to_csv(vali_file_name, header=True, index = False, encoding='cp949')    


def fn_create_word2vec(data_df, w2v_model_name):
    okt = Okt()
    review_data = data_df['review'].values
    pos_list = ['Adjective', 'Adverb', 'Determiner', 'Noun', 'Number', 'Verb']
    data = [okt.pos(xx, norm=True, stem=True) for xx in tqdm(review_data)]
    second_data = []
    for idx in range(len(data)):
        second_data.append([ i for i, y in data[idx] if y in pos_list ])    
        
    w2v_model = Word2Vec(second_data, size=10, window=5, min_count=10, sg=1)
    w2v_model.init_sims(replace=True)
 
    return w2v_model


if __name__ == "__main__":
    train_file_name = 'hw_4_train_data_KJY.csv'
    vali_file_name = 'hw_4_validation_data_KJY.csv'
    test_file_name = 'hw_4_test_data_KJY.csv'
    
    w2v_model_name = 'hw_4_word2vec_KJY.model'
    
    os.chdir(r'C:\Users\ggy01\OneDrive\바탕 화면\BIZ LAB\project4')
    data_file_name = 'hw_4_data.csv'
    data_df = pd.read_csv(data_file_name, encoding='cp949')[['label', 'review']]
    print('data_df shape - ', data_df.shape)
    
    data_df = data_df.drop_duplicates()
    print('data_df(drop_duplicates) shape - ', data_df.shape)

    
    print(data_df.groupby(['label'])['label'].count())

# =============================================================================

    os.chdir(r'C:\Users\ggy01\OneDrive\바탕 화면\BIZ LAB\project4')
    fn_data_split(data_df, train_file_name, vali_file_name, test_file_name)
    w2v_model = fn_create_word2vec(data_df, w2v_model_name)
#    w2v_model = Word2Vec.load(w2v_model_name)
    
    for test_word in ['월요일' , '배송', '빠르다', '좋다', '감사', '별로']:
        print('*'*50 + '\n' + test_word)
        pprint(w2v_model.wv.most_similar(test_word, topn=5))

            

'''

**************************************************
월요일
[('금요일', 0.959784746170044),
 ('목요일', 0.956427276134491),
 ('수요일', 0.9455667734146118),
 ('화요일', 0.9429277777671814),
 ('토욜', 0.9383894801139832)]
**************************************************
배송
[('파른', 0.7278835773468018),
 ('리오네', 0.7141883373260498),
 ('송도', 0.7126674652099609),
 ('명절', 0.6764638423919678),
 ('송이', 0.6738250255584717)]
**************************************************
빠르다
[('감솨', 0.7879383563995361),
 ('파른', 0.7672903537750244),
 ('총알', 0.7666589021682739),
 ('빨랏', 0.7610298991203308),
 ('빨르다', 0.7538026571273804)]
**************************************************
좋다
[('잘삿어', 0.7105856537818909),
 ('정말로', 0.7042036056518555),
 ('욧', 0.7030841708183289),
 ('아영', 0.7001270055770874),
 ('좋아욤', 0.6972665786743164)]
**************************************************
감사
[('하비다', 0.7892617583274841),
 ('감솨', 0.7831730842590332),
 ('고맙다', 0.7772567272186279),
 ('감사하다', 0.7344866394996643),
 ('힙니', 0.7170889973640442)]
**************************************************
별로
[('별루', 0.6891950368881226),
 ('그닥', 0.6276479959487915),
 ('역다', 0.5523307919502258),
 ('이외', 0.5468701124191284),
 ('안좋다', 0.5464729070663452)]

'''
