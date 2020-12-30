from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re, time, os, csv


def fn_webcrawler(cw, category, item_id, page=0):
    global crawling_count
    while 1:
        page += 1
        url = 'http://deal.11st.co.kr/product/SellerProductDetail.tmall?method=getProductReviewList&prdNo={}&page={}'.format(item_id, page)
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
       
        data_dict = {}
        soup_data = soup.find_all('div', 'selr_wrap')
        reviews = soup.find_all('p', 'bbs_summary')
        
        for idx in range(0, len(soup_data)):
            scores = soup_data[idx].find(string=re.compile('별5개 중'))
            if scores == None:
                continue

            data_dict[idx] = re.findall('\s\d', scores)
            data_dict[idx] = [int(i) for i in data_dict[idx]]
            if data_dict[idx][0] == 4 or data_dict[idx][0] == 5:
                data_dict[idx].append(1)
            elif data_dict[idx][0] == 1 or data_dict[idx][0] == 2:
                data_dict[idx].append(-1)
            else:
                data_dict[idx].append(0)
            data_dict[idx].insert(0, page)
            data_dict[idx].insert(0, item_id)
            data_dict[idx].insert(0, category)
            review = reviews[idx].text.strip()
            review = review.encode("cp949", "ignore")
            review = review.decode('cp949')
            data_dict[idx].append(review)
            if reviews[idx].text.strip() == '':
                del data_dict[idx]          
                 
        if len(soup_data) == 0 :
            break
        data_list = []
        for yy in data_dict.values():
            data_list.append(yy)
        cw.writerows(data_list)
                
        time.sleep(0.01)
        

if __name__ == "__main__":
    start_time = time.time()
    
    crawling_count = 0
    
    save_file_path = r'C:\Users\ggy01\OneDrive\바탕 화면\BIZ LAB\project3'
    os.chdir(save_file_path)
    save_file_name = 'hw_3_webcrawling_KJY.csv'
    
    item_list = [('chair', 1815000878), ('chair', 12423596), ('chair', 87595509), ('chair', 10843324), ('chair', 218190216), ('chair', 12623125) ,('chair', 50942984), ('char', 374683075)]
    
    with open(save_file_name, 'w', newline='', encoding='cp949') as f:
        cw = csv.writer(f)
        cw.writerow(['category', 'item_id', 'page', 'score', 'sentiment', 'review'])
        
        for i, x in enumerate(item_list):
            category, item_id = x
            print('{0}\n{1} / {2}\n{3} - {4}\n{0}'.format('-'*100, i+1, len(item_list), category, item_id))
            fn_webcrawler(cw, category, item_id)
            tmp_running_time = time.time() - start_time
            print('%s\ntmp running time : %d m  %0.2f s\n%s\n'%('#'*100, tmp_running_time//60, tmp_running_time%60, '#'*100))
    
    running_time = time.time() - start_time
    print('%s\ntotal running time : %d m  %0.2f s\n%s'%('#'*100, running_time//60, running_time%60, '#'*100))
            
    crawling_data = pd.read_csv(save_file_name, encoding='cp949')
    print(crawling_data.shape)
    print(crawling_data.head())
