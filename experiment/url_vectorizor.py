import numpy as np
import pandas as pd
from urllib.parse import urlparse
from tld import get_tld
import re
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action="store", dest="data_path", type=str,
                    required=False, help="path of the dataset", default="./data/raw/url_data.csv")

parser.add_argument('--out_path', action="store", dest="out_path", type=str,
                    required=False, help="path of the output", default="./data/vectorized/")

args = parser.parse_args()
data_path = args.data_path
out_path = args.out_path

# Load data
print("reading file")
data = pd.read_csv(data_path)
# remove unlabled entries
print("removing unknown labels")
data = data.drop('Unnamed: 0', axis=1)

print("vectorizing urls")
# Length of URL
data['url_length'] = data['url'].apply(lambda i: len(str(i)))

# Hostname Length
data['hostname_length'] = data['url'].apply(lambda i: len(urlparse(i).netloc))

# Path Length
data['path_length'] = data['url'].apply(lambda i: len(urlparse(i).path))

# First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

data['fd_length'] = data['url'].apply(lambda i: fd_length(i))


# Length of Top Level Domain
data['tld'] = data['url'].apply(lambda i: get_tld(i,fail_silently=True))
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

data['tld_length'] = data['tld'].apply(lambda i: tld_length(i))

data = data.drop("tld",1)
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

data['count-'] = data['url'].apply(lambda i: i.count('-'))
data['count@'] = data['url'].apply(lambda i: i.count('@'))
data['count?'] = data['url'].apply(lambda i: i.count('?'))
data['count%'] = data['url'].apply(lambda i: i.count('%'))
data['count.'] = data['url'].apply(lambda i: i.count('.'))
data['count='] = data['url'].apply(lambda i: i.count('='))
data['count-http'] = data['url'].apply(lambda i : i.count('http'))
data['count-https'] = data['url'].apply(lambda i : i.count('https'))
data['count-www'] = data['url'].apply(lambda i: i.count('www'))
data['count-digits']= data['url'].apply(lambda i: digit_count(i))
data['count-letters']= data['url'].apply(lambda i: letter_count(i))
data['count_dir'] = data['url'].apply(lambda i: no_of_dir(i))

#Use of IP or not in domain
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return -1
    else:
        # print 'No matching pattern found'
        return 1

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return -1
    else:
        return 1
    
data['use_of_ip'] = data['url'].apply(lambda i: having_ip_address(i))
data['short_url'] = data['url'].apply(lambda i: shortening_service(i))

print("writing to file")

data.to_csv(f"{out_path}/url_data_vectorized.csv", index=False)
