# coding=utf-8
'''
    A demo of web image crawler that grabs all jpg files from a URL provided by the user and stores them in OSS.
    For example：invk crawler_image -s "{\"url\":\"http://www.xxxx.com\"}"
    Please replace <' var '> with real value 
'''

import re
import urllib
import json
import datetime
import oss2
import logging


# The main function
def handler(environ, start_response):
    logger = logging.getLogger()
    logger.info('start worker')
    try:
        request_body_size = int(environ.get('CONTENT_LENGTH', 0))
    except (ValueError):
        request_body_size = 0
    request_body = environ['wsgi.input'].read(request_body_size)
    logger.info(request_body)
    
    context = environ['fc.context']
    evt = json.loads(request_body)
    url = evt['url']
    logger.info('url:' + url)
    endpoint = 'oss-eu-central-1.aliyuncs.com'
    creds = context.credentials
    auth = oss2.StsAuth(creds.accessKeyId, creds.accessKeySecret, creds.securityToken)
    bucket = oss2.Bucket(auth, endpoint, 'chmurowiskophotos')

    html = getHtml(url)
    img_list = getImg(html)
    count = 0
    for item in img_list:
        count += 1
        logging.info(item)
        # Get each picture
        pic = urllib.urlopen(item)
        # Store all the pictures in oss bucket, keyed by timestamp in microsecond unit
        bucket.put_object(str(datetime.datetime.now().microsecond) + '.jpg', pic)

    status = '200 OK'
    response_headers = [('Content-type', 'text/plain')]
    start_response(status, response_headers)
    response = "download success, total pictures:" + str(count)
    #return ['download success, total pictures:' + str(count)]
    return [bytes(response)]


# get url of content
def getHtml(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html


# Get jpg uri
def getImg(html):
    reg = r'http:\/\/[^\s,"]*\.jpg'
    imgre = re.compile(reg)
    imglist = re.findall(imgre, html)
    logger = logging.getLogger()
    logger.info(imglist)
    return imglist