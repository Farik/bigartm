import csv
import urllib.request
import sys
import time
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("NTAutils")
logger.setLevel(logging.DEBUG)


def get_document_text(url):
    print("going to download %s" % url)
    start_time = time.time()
    text = urllib.request.urlopen(url).read()
    elapsed_time = time.time() - start_time
    print('document {} size=[{}] downloaded in {}ms'.format(url, sys.getsizeof(text), int(elapsed_time * 1000)))
    return text


def cache_document_text_on_disk(limit=100):
    line = 0
    print('ready to download files')
    with open("documents/_sources.csv", 'r') as csvfile:
        print('read document list')
        document_description = csv.reader(csvfile)
        for row in document_description:
            if line > limit:
                break
            [serverCode, documentId, categoryId, textUrl] = row
            with open("documents/"+documentId, "wb") as text_file:
                safe_url = textUrl[:9]+"deploy"+textUrl[9:]
                text_file.write(get_document_text(safe_url))
            line += 1

if __name__ == "__main__":
    cache_document_text_on_disk(30000)
