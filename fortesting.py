# import numpy as np
#
# s = "testing"
# print(type(s))
# print(s[0])
#
# file = open('test','a')
# file.write("this is for testing")
#
# file.close()

# -*- coding:utf-8 -*-
import urllib.request as u
from bs4 import BeautifulSoup
import socket
import http.client as hc


class Spider(object):
    """Spider"""

    def __init__(self, url):
        self.url = url

    def getNextUrls(self):
        urls = []
        request = u.Request(self.url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; \
            WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36')
        try:
            html = u.urlopen(request)
        except socket.timeout as e:
            pass
        except u.URLError as ee:
            pass
        except hc.BadStatusLine:
            pass

        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a'):
            print("http://m.sohu.com" + link.get('href'))
            if link.get('href')[0] == '/':
                urls.append("http://m.sohu.com" + link.get('href'))
        return urls


def getNews(url):
    # print(url)
    xinwen = ''
    request = u.Request(url)
    request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; \
        WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36')
    try:
        html = u.urlopen(request)
    except u.HTTPError as e:
        print(e.code)

    soup = BeautifulSoup(html, 'html.parser')
    for news in soup.select('p.para'):
        xinwen += news.get_text().decode('utf-8')
    return xinwen


class News(object):
    """
    source:from where 从哪里爬取的网站
    title:title of news  文章的标题
    time:published time of news 文章发布时间
    content:content of news 文章内容
    type: type of news    文章类型
    """

    def __init__(self, source, title, time, content, type):
        self.source = source
        self.title = title
        self.time = time
        self.content = content
        self.type = type



file = open('test.txt', 'a')
for i in range(38, 50):
    for j in range(1, 5):
        url = "http://m.sohu.com/cr/" + str(i) + "/?page=" + str(j)
        print(url)
        s = Spider(url)
        for newsUrl in s.getNextUrls():
            file.write(getNews(newsUrl))
            file.write("\n")
            print("---------------------------")
































