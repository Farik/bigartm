import csv
#import urllib.request
import sys
import time
import logging


from twisted.internet import reactor, threads
from urlparse import urlparse
import httplib
import itertools

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("NTAutils")
logger.setLevel(logging.DEBUG)


# def get_document_text(url):
#     print("going to download %s" % url)
#     start_time = time.time()
#     text = urllib.request.urlopen(url).read()
#     elapsed_time = time.time() - start_time
#     print('document {} size=[{}] downloaded in {}ms'.format(url, sys.getsizeof(text), int(elapsed_time * 1000)))
#     return text
#
#
# def cache_document_text_on_disk(limit=100):
#     line = 0
#     print('ready to download files')
#     with open("documents/_sources.csv", 'r') as csvfile:
#         print('read document list')
#         document_description = csv.reader(csvfile)
#         for row in document_description:
#             if line > limit:
#                 break
#             [serverCode, documentId, categoryId, textUrl] = row
#             with open("documents/"+documentId, "wb") as text_file:
#                 safe_url = textUrl[:9]+"deploy"+textUrl[9:]
#                 text_file.write(get_document_text(safe_url))
#             line += 1


def save_document(id, content):
    with open("documents_10/"+id, "wb") as text_file:
        text_file.write(content)


def getStatus(ourl, id):
    url = urlparse(ourl)
    conn = httplib.HTTPConnection(url.netloc)
    conn.request("GET", url.path+"?"+url.query)
    res = conn.getresponse()
    if res.status == 200:
        save_document(id, res.read())
    return res.status


def processResponse(response, url,):
    print response, url
    processedOne()


def processError(error,url):
    print "error", url#, error
    processedOne()


def processedOne():
    if finished.next() == added:
        reactor.stop()


def addTask(url, id):
    req = threads.deferToThread(getStatus, url, id)
    req.addCallback(processResponse, url)
    req.addErrback(processError, url)

concurrent = 200
finished = itertools.count(1)
added = None


def cache_document_text_on_disk_much_faster(limit=100):

    reactor.suggestThreadPoolSize(concurrent)
    global added
    added = 0
    with open("documents_10/_sources.csv", 'r') as csvfile:
        print('read document list')
        document_description = csv.reader(csvfile)
        for row in document_description:
            if added > limit:
                break
            [serverCode, documentId, categoryId, textUrl] = row
            safe_url = textUrl
            if "flashcards" not in textUrl:
                safe_url = textUrl[:9]+"deploy"+textUrl[9:]

            added += 1
            addTask(safe_url, documentId)

    try:
        reactor.run()
    except KeyboardInterrupt:
        reactor.stop()

if __name__ == "__main__":
    cache_document_text_on_disk_much_faster(3000)
