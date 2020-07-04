import cv2
from multiprocessing import Process
import os
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from urllib.request import urlretrieve
from urllib.request import urlopen
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from functools import partial
from time import strftime
from random import choice
import string

import sys

    




def imageDown(path,url):
    cap = cv2.VideoCapture(url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    i=0
    arr = [choice(string.ascii_letters) for _ in range(10)] # 난수 8글자 랜덤으로 지정
    pid = ''.join(arr)
    try:
        if not os.path.isdir('database'+pid):
            os.makedirs('database/'+pid)
    except:
        pass
    while True:
        i+=1
        ret,frame = cap.read()
        if ret is False:
            break
        if i % round(1500/fps) == 0:
            cv2.imwrite('database/'+pid+'/' + path+str(i)+'.jpg',frame)
    cap.release()

def bo_url():
    urlmap = []

    for k in range(1,17):
        URL = ''+str(k)
        res = requests.get(URL)
        bs = BeautifulSoup(res.text, 'html.parser')
        for i in set(bs.select('div > a')):
            if 'wr_id' in i.get('href') and i.text is not '\n\n' and i.get('href') not in urlmap:
                urlmap.append(i.get('href'))    
    return urlmap
def bo_video(links):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    driver = webdriver.Chrome('chromedriver.exe')
    videomap=[]
    driver.get(links)
    title = driver.find_element_by_id('bo_v_title').text
    iframe=driver.find_elements_by_tag_name('iframe')[0]
    driver.switch_to_frame(iframe)
    try:
        videomap.append([title,driver.find_element_by_xpath('/html/body/div[1]/video/source[@id="video_source"]').get_attribute('src')])
    except:
        videomap.append([title,'None'])
    driver.close()
    return videomap    

def DD_url():
    urlmap1 = []

    for k in range(1,80):
        URL = ''+str(k)
        res = requests.get(URL)
        bs = BeautifulSoup(res.text, 'html.parser')
        for i in set(bs.select('.item-subject')):
            urlmap1.append(i.get('href'))
    return urlmap1

def DD_video(links):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    driver = webdriver.Chrome('chromedriver.exe')
    videomap=[]
    driver.get(links)
    ad = WebDriverWait(driver,300).until(EC.presence_of_element_located((By.XPATH, '//*[@id="movie_player"]/iframe')))
    title = driver.find_element_by_xpath('//*[@id="thema_wrapper"]/div[6]/div/div/div[1]/div[3]/section/article/div/h1').text
    iframe=  driver.find_element_by_xpath('//*[@id="movie_player"]/iframe')
    video = iframe.get_attribute('src')
    try:
        videomap.append([title,video])
    except:
        videomap.append([title,'None'])
    driver.close()            
    return videomap    


if __name__ == '__main__':
    CPU_CORE = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=CPU_CORE)
    videomap = pool.map(DD_video,DD_url())
    videomap1 = pool.map(bo_video,bo_url())
    pool.close()
    video = videomap + videomap1
    
    for i in range(0,len(video),6):
        th1 = Process(target=imageDown, args=(video[i][0][0],video[i][0][1]))
        th1.start()
        if i+1 < len(video):
            th2 = Process(target=imageDown, args=(video[i+1][0][0],video[i+1][0][1]))
            th2.start()
        if i+2 < len(video):
            th3 = Process(target=imageDown, args=(video[i+2][0][0],video[i+2][0][1]))
            th3.start()
        if i+3 < len(video):
            th4 = Process(target=imageDown, args=(video[i+3][0][0],video[i+3][0][1]))
            th4.start()
        if i+4 < len(video):
            th5 = Process(target=imageDown, args=(video[i+3][0][0],video[i+4][0][1]))
            th5.start()
        if i+5 < len(video):
            th6 = Process(target=imageDown, args=(video[i+3][0][0],video[i+5][0][1]))
            th6.start()
        if th1:
            th1.join()
        if th2:
            th2.join()
        if th3:
            th3.join()
        if th4:
            th4.join()
        if th5:
            th5.join()
        if th6:
            th6.join()


