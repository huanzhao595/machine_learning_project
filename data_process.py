import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import numpy as np
import pandas as pd
import random

import chardet

# upzip the file: gunzip -k weird.log.gz

# URL of the file to be downloaded
# url = "http://www.secrepo.com/maccdc2012/dns.log.gz"
# url = "http://www.secrepo.com/maccdc2012/files.log.gz"
# url = "http://www.secrepo.com/maccdc2012/ftp.log.gz"
# url = "http://www.secrepo.com/maccdc2012/http.log.gz"
# url = "http://www.secrepo.com/maccdc2012/notice.log.gz"
# url = "http://www.secrepo.com/maccdc2012/signatures.log.gz"
# url = "http://www.secrepo.com/maccdc2012/smtp.log.gz"
# url = "http://www.secrepo.com/maccdc2012/ssh.log.gz"
# url = "http://www.secrepo.com/maccdc2012/ssl.log.gz"
# url = "http://www.secrepo.com/maccdc2012/tunnel.log.gz"
# url = "http://www.secrepo.com/maccdc2012/weird.log.gz"

# Local file path where the downloaded file will be saved
# local_file_path = "./datasciense/data/dns.log.gz"
# local_file_path = "./datasciense/data/files.log.gz"
# local_file_path = "./datasciense/data/ftp.log.gz"
# local_file_path = "./datasciense/data/http.log.gz"
# local_file_path = "./datasciense/data/notice.log.gz"
# local_file_path = "./datasciense/data/signatures.log.gz"
# local_file_path = "./datasciense/data/smtp.log.gz"
# local_file_path = "./datasciense/data/ssh.log.gz"
# local_file_path = "./datasciense/data/ssl.log.gz"
# local_file_path = "./datasciense/data/tunnel.log.gz"
# local_file_path = "./datasciense/data/weird.log.gz"

# session = Session()

# retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
# adapter = HTTPAdapter(max_retries=retries)

# session.mount('http://', adapter)
# session.mount('https://', adapter)

# with session.get(url, stream=True) as resp:
#     if resp.status_code == 200:
#         # Open the file in binary mode and write the content of the response to it
#         with open(local_file_path, 'wb') as file:
#             for chunk in resp.iter_content(chunk_size=8192):
#                 file.write(chunk)
#         print(f"File has been downloaded and saved to {local_file_path}")
#     else:
#         print(f"Failed to retrieve the file. Server responded with: {resp.status_code} - {resp.reason}")

# Send a HTTP request to the server and save
# the HTTP response in a response object called resp
# with requests.get(url, stream=True) as resp:
#     # Check if the request was successful
#     if resp.status_code == 200:
#         # Open the file in binary mode and write the content of the response to it
#         with open(local_file_path, 'wb') as file:
#             for chunk in resp.iter_content(chunk_size=8192):
#                 file.write(chunk)
#         print(f"File has been downloaded and saved to {local_file_path}")
#     else:
#         print(f"Failed to retrieve the file. Server responded with: {resp.status_code} - {resp.reason}")

# load log and store in dat
filename = "./datasciense/data/dhcp.log" # (1502, 10)
# filename = "./datasciense/data/dns.log" # (427935, 23)
# filename = "./datasciense/data/files.log" # (839716, 23)
# filename = "./datasciense/data/ftp.log" # (5796, 19)

# filename = "./datasciense/data/http.log" # too much data can't load 
# split -l 500000 -d --additional-suffix=.log ./datasciense/data/http.log ./datasciense/data/http_
# filename = "./datasciense/data/http_00.log" # (499056, 27)
# filename = "./datasciense/data/http_01.log" # (500000, 27)
# filename = "./datasciense/data/http_02.log" # (499994, 27)
# filename = "./datasciense/data/http_03.log" # (500000, 27)
# filename = "./datasciense/data/http_04.log" # (48410, 27)

# filename = "./datasciense/data/notice.log" #  (682, 26)
# filename = "./datasciense/data/signatures.log" # (1, 11)
# filename = "./datasciense/data/smtp.log" # (194, 25)
# filename = "./datasciense/data/ssh.log" # (7143, 15)
#filename = "./datasciense/data/ssl.log" # (56214, 19)
#filename = "./datasciense/data/tunnel.log" # (280, 8)
#filename = "./datasciense/data/weird.log" # (65983, 10)

filename1 = "./datasciense/data/dhcp.dat"
# filename1 = "./datasciense/data/dns.dat"
# filename1 = "./datasciense/data/files.dat"
# filename1 = "./datasciense/data/ftp.dat"

# filename1 = "./datasciense/data/http.dat"
# filename1 = "./datasciense/data/http_00.dat" 
# filename1 = "./datasciense/data/http_01.dat" 
# filename1 = "./datasciense/data/http_02.dat"
# filename1 = "./datasciense/data/http_03.dat"
# filename1 = "./datasciense/data/http_04.dat"

# filename1 = "./datasciense/data/notice.dat"
# filename1 = "./datasciense/data/signatures.dat"
# filename1 = "./datasciense/data/smtp.dat"
# filename1 = "./datasciense/data/ssh.dat"
#filename1 = "./datasciense/data/ssl.dat"
#filename1 = "./datasciense/data/tunnel.dat"
#filename1 = "./datasciense/data/weird.dat"

# data = pd.read_csv(filename, sep='\t', header=None)
# data = pd.read_csv(filename, sep='\t', error_bad_lines=False, header=None)

# print(data.head())
# print(data.shape)
# data.columns = ['UserID', 'ItemID', 'Rating', 'TimeStamp']

# data.to_csv(filename1, sep=',', index=False, header=None)

# load Excel
df = pd.read_excel('./datasciense/data/conn.xlsx', engine='openpyxl') # (197505, 20)
print(df.head())
print(df.shape)
# store DataFrame as .dat
df.to_csv('./datasciense/data/conn.dat', index=True, sep='\t') 