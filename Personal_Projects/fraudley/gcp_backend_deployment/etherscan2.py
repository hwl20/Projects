import os
import requests
import numpy as np
import pandas as pd 
from datetime import datetime

ETHER_VALUE = 10 ** 18
API_KEY = "SVFQMQGPZ6NGU7GD5E281BGCZX1RTKU4Y5"
API_KEY_1 = "KN3BXQGZF1XH7VMPMHU72HG1XA4KCF21E5"
API_KEY_2 = "PNED3EGX52WZQPJRT3TVC77DGSAVJQ1UKK"
API_KEY_3 = "R96YPW4Z5MFNZAAHQZ4YVJ3SGVK7EFKM3E"
API_KEY_4 = "SWAZXYDTDP7HB9FBNVP8UGGHVHHB7RGB2F"
API_KEY_5 = "DGHKQQ11ESG5FUEG89CY4232QQFJJBUPGK"
API_KEY_6 = "7HWX4UZ14IPQYB2K4X3M3AVRRXWV4D8ZNW"
API_KEY_7 = "HSR4RNQSJD8YRKNCX1VBDTTK455TJC91W6"
API_KEY_8 = "URD76H4A6EMNM4VXUGWPNPZ51KIV88573J"
API_KEY_9 = "SVYQUE74NDZFI2XMHS66H5R97JY72YAEJ2"

base_url = "https://api.etherscan.io/api"

def balance_checker(address):
    url = f'https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={API_KEY_1}'
    try:
        response = requests.request('GET', url)
        if response.status_code != 200:
            url2 = f'https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={API_KEY_1}'
            response2 = requests.request('GET', url2)
            if response2.status_code != 200:
                balance = -10**20
                print(f'Error for getting address balance: {address}')
            else:
                balance = response2.json()['result']
        else:
            balance = response.json()['result']
    except:
        balance = -10**20
    return float(balance) / ETHER_VALUE

def get_normal_transactions_address(address,
                                    start_block=0,
                                    end_block=99999999,
                                    page=1,
                                    offset=10000,
                                    sort="asc"):
    assert sort in ["asc", "desc"]
    url = f"https://api.etherscan.io/api" \
          f"?module=account" \
          f"&action=txlist" \
          f"&address={address}" \
          f"&startblock={start_block}" \
          f"&endblock={end_block}" \
          f"&page={page}" \
          f"&offset={offset}" \
          f"&sort={sort}" \
          f"&apikey={API_KEY_2}"
    response = requests.get(url, timeout=1000)
    if response.status_code == 200:
        return response.json()
    return None

def get_normal_transactions_address2(address,
                                    start_block=0,
                                    end_block=99999999,
                                    page=1,
                                    offset=10000,
                                    sort="asc"):
    assert sort in ["asc", "desc"]
    url = f"https://api.etherscan.io/api" \
          f"?module=account" \
          f"&action=txlist" \
          f"&address={address}" \
          f"&startblock={start_block}" \
          f"&endblock={end_block}" \
          f"&page={page}" \
          f"&offset={offset}" \
          f"&sort={sort}" \
          f"&apikey={API_KEY_3}"
    response = requests.get(url, timeout=1000)
    if response.status_code == 200:
        return response.json()
    return None

def get_normal_transactions_address3(address,
                                    start_block=0,
                                    end_block=99999999,
                                    page=1,
                                    offset=10000,
                                    sort="asc"):
    assert sort in ["asc", "desc"]
    url = f"https://api.etherscan.io/api" \
          f"?module=account" \
          f"&action=txlist" \
          f"&address={address}" \
          f"&startblock={start_block}" \
          f"&endblock={end_block}" \
          f"&page={page}" \
          f"&offset={offset}" \
          f"&sort={sort}" \
          f"&apikey={API_KEY_4}"
    response = requests.get(url, timeout=1000)
    if response.status_code == 200:
        return response.json()
    return None

def get_minmax_ether_sentReceived_by_address(address):
    page = 1
    resp = get_normal_transactions_address(address, page=page)
    res_min_sent = None
    res_max_sent = None
    res_min_received = None
    res_max_received = None
    try:
        while resp is not None and resp['status'] == '1':
            for r in resp['result']:
                if r['from'] == address.lower():
                    if (res_min_sent is None) or (res_max_sent is None):
                        res_min_sent = int(r['value'])
                        res_max_sent = int(r['value'])
                    else:
                        res_min_sent = min(res_min_sent, int(r['value']))
                        res_max_sent = max(res_max_sent, int(r['value']))
                
                if r['to'] == address.lower():
                    if (res_min_received is None) or (res_max_received is None):
                        res_min_received = int(r['value'])
                        res_max_received = int(r['value'])
                    else:
                        res_min_received = min(res_min_received, int(r['value']))
                        res_max_received = max(res_max_received, int(r['value']))           
            page += 1
            resp = get_normal_transactions_address(address, page=page)
    except:
        return -10**20, -10**20, -10**20, -10**20
    res_max_sent = 0 if res_max_sent == None else res_max_sent
    res_min_sent = 0 if res_min_sent == None else res_min_sent
    res_max_received = 0 if res_max_received == None else res_max_received
    res_min_received = 0 if res_min_received == None else res_min_received
    return res_max_sent/ETHER_VALUE, res_min_sent/ETHER_VALUE, res_max_received/ETHER_VALUE, res_min_received/ETHER_VALUE

def make_api_url(module, action, address, **kwargs):
	url = base_url + f"?module={module}&action={action}&address={address}&apikey={API_KEY_5}"

	for key, value in kwargs.items():
		url += f"&{key}={value}"

	return url

def make_api_url2(module, action, address, **kwargs):
	url = base_url + f"?module={module}&action={action}&address={address}&apikey={API_KEY_6}"

	for key, value in kwargs.items():
		url += f"&{key}={value}"

	return url

def make_api_url3(module, action, address, **kwargs):
	url = base_url + f"?module={module}&action={action}&address={address}&apikey={API_KEY_7}"

	for key, value in kwargs.items():
		url += f"&{key}={value}"

	return url

def make_api_url4(module, action, address, **kwargs):
	url = base_url + f"?module={module}&action={action}&address={address}&apikey={API_KEY_8}"

	for key, value in kwargs.items():
		url += f"&{key}={value}"

	return url

def make_api_url5(module, action, address, **kwargs):
	url = base_url + f"?module={module}&action={action}&address={address}&apikey={API_KEY_9}"

	for key, value in kwargs.items():
		url += f"&{key}={value}"

	return url

def get_time_diff(address):
    try:    
        transactions_url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
        response = requests.get(transactions_url)
        data = response.json()["result"]

        internal_tx_url = make_api_url("account", "txlistinternal", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
        response2 = requests.get(internal_tx_url)
        data2 = response2.json()["result"]

        data.extend(data2)
        data.sort(key=lambda x: int(x['timeStamp']))


        if len(data) == 0:
            d = 0
        else:
            time_diff_unix = int(data[-1]["timeStamp"]) - int(data[0]["timeStamp"])

            time_diff = time_diff_unix/60
            d = time_diff
    except:
        d = 0
    
    return d

def get_mean_time_btw_received(address):
    try:
        transactions_url = make_api_url2("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
        response = requests.get(transactions_url)
        data = response.json()["result"]

        internal_tx_url = make_api_url2("account", "txlistinternal", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
        response2 = requests.get(internal_tx_url)
        data2 = response2.json()["result"]

        data.extend(data2)
        data = list(filter(lambda x: x["to"] == address, data))
        data.sort(key=lambda x: int(x['timeStamp']))
        
        time_list = []
        if len(data) == 0:
            d = 0
        elif len(data) == 1:
            d = int(data[0]['timeStamp'])/60
        else:
            for i in range(1,len(data)):
                time_list.append((int(data[i]['timeStamp'])-int(data[i-1]['timeStamp']))/ 60)

            d = sum(time_list) / len(time_list)
    except:
        d = 0
            
    return d

#print(get_mean_time_btw_received(addresses))
  
def get_mean_time_btw_sent(address):
    try:
        transactions_url = make_api_url3("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
        response = requests.get(transactions_url)
        data = response.json()["result"]

        internal_tx_url = make_api_url3("account", "txlistinternal", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
        response2 = requests.get(internal_tx_url)
        data2 = response2.json()["result"]
        
        data.extend(data2)
        data = list(filter(lambda x: x["from"] == address, data))
        data.sort(key=lambda x: int(x['timeStamp']))
        
        time_list = []
        if len(data) == 0:
            d = 0
        elif len(data) == 1:
            d = int(data[0]['timeStamp'])/60
        else:
            for i in range(1,len(data)):
                time_list.append((int(data[i]['timeStamp'])-int(data[i-1]['timeStamp']))/ 60)

            d = sum(time_list) / len(time_list)
    except:
        d = 0
            
    return d

# Get total transactions, sent transactions, received transactions of a wallet.
# Rationale of -(10**7) is that if we have an ERROR, the number of transactions will be < 0, indicating an ERROR of this function
def get_txn_statistics(addr, startblock = 0):
    to_discard = True
    url = f'https://api.etherscan.io/api?module=account&action=txlist&startblock={startblock}&address={addr}&apikey={API_KEY}'
    response1 = requests.request('GET', url)
    if (response1.status_code != 200):
        txn_count = -(10**7)
        sent_txn_count = -(10**7)
        received_txn_count = -(10**7)
        print(f'Error for getting txn address: {addr}')
    else:
        txn_info = response1.json()['result']
        try:
            if txn_info == []:
                txn_count = 0
                sent_txn_count = 0
                received_txn_count = 0 
                unique_received_from = 0
                unique_sent_to = 0

            elif (type(txn_info)) == list:
                txn_count = len(txn_info)
                if any([x['from'] == x['to'] for x in txn_info]):
                    to_discard = False
                from_lst = [x['from'].lower() for x in txn_info]
                sent_txn_count = from_lst.count(addr)
                receieved_from_set = set(from_lst)
                if to_discard:
                    receieved_from_set.discard(addr)
                unique_received_from = len(receieved_from_set)

                to_lst = [x['to'].lower() for x in txn_info]
                received_txn_count = to_lst.count(addr)
                sent_to_set = set(to_lst)
                if to_discard:
                    sent_to_set.discard(addr)
                unique_sent_to = len(sent_to_set)
            else:
                txn_count = -(10**7)
                sent_txn_count = -(10**7)
                received_txn_count = -(10**7)
                unique_received_from = -(10**7)
                unique_sent_to = -(10**7)
        except:
            txn_count = -(10**7)
            sent_txn_count = -(10**7)
            received_txn_count = -(10**7)
            unique_received_from = -(10**7)
            unique_sent_to = -(10**7)            
            print(f'Error in respsonse of API call for address: {addr}')          
    return txn_count, sent_txn_count, received_txn_count, unique_received_from, unique_sent_to

def get_total_erc20_txns(address):
    transaction_url = make_api_url4("account", "tokentx", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transaction_url)
    data = response.json()["result"]
    return len(data)

def get_erc20_received_sent(address):
    received_dict = {}
    sent_dict = {} 
    transactions_url = make_api_url5("account", "tokentx", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")

    try:
        response = requests.get(transactions_url)
        data = response.json()["result"]
        if (data):
            for txn in data:
                if (txn):
                    if txn["from"] == address:
                        if txn["to"] not in sent_dict:
                            sent_dict[txn["to"]] = 1
                    if txn["to"] == address:
                        if txn["from"] not in received_dict:
                            received_dict[txn["from"]] = 1 
    except:
        received_dict = {}
        sent_dict = {} 
    return len(received_dict), len(sent_dict)

def get_total_average_ether_received_by_address(address):
    page = 1
    resp = get_normal_transactions_address2(address, page=page)
    total = 0
    count = 0
    while resp is not None and resp['status'] == '1':
        for r in resp['result']:
            if r['to'] == address.lower():
                total += int(r['value'])
                count += 1
        page += 1
        resp = get_normal_transactions_address2(address, page=page)
    if count == 0:
        return 0, 0
    return total/ETHER_VALUE, (total/count)/ETHER_VALUE

def get_total_average_ether_sent_by_address(address):
    page = 1
    resp = get_normal_transactions_address3(address, page=page)
    total = 0
    count = 0
    while resp is not None and resp['status'] == '1':
        for r in resp['result']:
            if r['from'] == address.lower():
                total += int(r['value'])
                count += 1
        page += 1
        resp = get_normal_transactions_address3(address, page=page)
    if count == 0:
        return 0, 0
    return total/ETHER_VALUE, (total/count)/ETHER_VALUE