"""Quick test to check which AVA annotation/video URLs are accessible."""
import urllib.request
import urllib.error
import sys

URLS_TO_TEST = [
    # Annotation files (S3 mirror)
    "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt",
    "https://s3.amazonaws.com/ava-dataset/annotations/ava_train_v2.2.csv",
    "https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip",
    # Annotation fallbacks
    "https://huggingface.co/datasets/Loie/AVA-dataset/resolve/main/annotations/ava_file_names_trainval_v2.1.txt",
    # Video files - verified working pattern: trainval/<id>.mkv
    "https://s3.amazonaws.com/ava-dataset/trainval/_-Z6wFjXtGQ.mkv",
    "https://s3.amazonaws.com/ava-dataset/trainval/-5KQ66BBWC4.mkv",
    "https://s3.amazonaws.com/ava-dataset/trainval/55Ihr6uVIDA.mkv",
    # HuggingFace mirror
    "https://huggingface.co/datasets/Loie/AVA-dataset/resolve/main/videos/_-Z6wFjXtGQ.mkv",
]

for url in URLS_TO_TEST:
    try:
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=15)
        size = resp.headers.get('Content-Length', 'unknown')
        print(f"OK  ({resp.status}) [{size} bytes] {url}")
    except urllib.error.HTTPError as e:
        # Some servers disallow HEAD, try ranged GET
        if e.code in (403, 405):
            try:
                req2 = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=0-0'})
                resp2 = urllib.request.urlopen(req2, timeout=15)
                print(f"OK  ({resp2.status}) [ranged] {url}")
            except Exception as e2:
                print(f"FAIL ({e.code}/{e2}) {url}")
        else:
            print(f"FAIL ({e.code}) {url}")
    except Exception as e:
        print(f"FAIL ({e}) {url}")
