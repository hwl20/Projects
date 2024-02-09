import os
# from xml.dom.expatbuilder import FragmentBuilder
import requests
import numpy as np
import pandas as pd 
from datetime import datetime

#df = pd.read_csv('fraud_n_non_fraud.csv')
ETHER_VALUE = 10 ** 18

#from dotenv import load_dotenv

#load_dotenv()
#API_KEY = os.getenv("API_KEY")



def get_balance_single_address(address,
                               tag="earliest"):
    """Returns the Ether balance of a given address
    :param address: the string representing the address to check for balance
    :param tag: the string pre-defined block parameter, either earliest, pending or latest
    """
    url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def balance_checker(address):
    url2 = f'https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={API_KEY}'
    response2 = requests.request('GET', url2)
    if response2.status_code != 200:
        balance = -10**20
        print(f'Error for getting address balance: {address}')
    else:
        balance = response2.json()['result']
    return float(balance) / ETHER_VALUE

def get_balance_multi_address(addresses,
                              tag="earliest"):
    """Returns the balance of the accounts from a list of addresses

    :param addresses: up to 20 addresses per call
    :param tag: the integer pre-defined block parameter, either earliest, pending or latest
    """
    assert tag in ["earliest", "pending", "latest"]
    addresses = ",".join(addresses)
    url = f"https://api.etherscan.io/api?" \
          f"module=account" \
          f"&action=balancemulti" \
          f"&address={addresses}" \
          f"&tag={tag}" \
          f"&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None


def get_normal_transactions_address(address,
                                    start_block=0,
                                    end_block=99999999,
                                    page=1,
                                    offset=10000,
                                    sort="asc"):
    """Returns the list of transactions performed by an address with optional pagination

    :param address: the string representing the addresses to check for balance
    :param start_block: the integer block number to start searching for transactions
    :param end_block: the integer block number to stop searching for transactions
    :param page: the integer page number, if pagination is enabled
    :param offset: the number of transactions displayed per page
    :param sort: the sorting preference, use asc to sort by ascending and desc to sort by descendin
    """
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
          f"&apikey={API_KEY}"
    response = requests.get(url, timeout=1000)
    if response.status_code == 200:
        return response.json()
    return None


def get_total_ether_received_by_address(address):
    page = 1
    resp = get_normal_transactions_address(address, page=page)
    total = 0
    while resp is not None and resp['status'] == '1':
        for r in resp['result']:
            if r['to'] == address.lower():
                total += int(r['value'])
        page += 1
        resp = get_normal_transactions_address(address, page=page)
    return total / ETHER_VALUE


def get_min_ether_received_by_address(address):
    page = 1
    resp = get_normal_transactions_address(address, page=page)
    res = None
    while resp is not None and resp['status'] == '1':
        for r in resp['result']:
            if r['to'] == address.lower():
                if res is None:
                    res = int(r['value'])
                else:
                    res = min(res, int(r['value']))
        page += 1
        resp = get_normal_transactions_address(address, page=page)
    return res / ETHER_VALUE


def get_max_ether_received_by_address(address):
    page = 1
    resp = get_normal_transactions_address(address, page=page)
    res = None
    while resp is not None and resp['status'] == '1':
        for r in resp['result']:
            if r['to'] == address.lower():
                if res is None:
                    res = int(r['value'])
                else:
                    res = max(res, int(r['value']))
        page += 1
        resp = get_normal_transactions_address(address, page=page)
    return res / ETHER_VALUE


def get_average_ether_received_by_address(address):
    page = 1
    resp = get_normal_transactions_address(address, page=page)
    total = 0
    count = 0
    while resp is not None and resp['status'] == '1':
        for r in resp['result']:
            if r['to'] == address.lower():
                total += int(r['value'])
                count += 1
        page += 1
        resp = get_normal_transactions_address(address, page=page)
    if count == 0:
        return 0
    return (total/count)/ETHER_VALUE


def get_total_ether_sent_by_address(address):
    page = 1
    resp = get_normal_transactions_address(address, page=page)
    total = 0
    count = 0
    try:
        while resp is not None and resp['status'] == '1':
            for r in resp['result']:
                if r['from'] == address.lower():
                    count += 1
                    total += int(r['value'])
                if count == 10000:
                    return total/ETHER_VALUE
            page += 1
            resp = get_normal_transactions_address(address, page=page)
    except:
        return -10**22
    return total/ETHER_VALUE


def get_min_ether_sent_by_address(address):
    page = 1
    resp = get_normal_transactions_address(address, page=page)
    res = None
    count = 0
    try:
        while resp is not None and resp['status'] == '1':
            for r in resp['result']:
                if r['from'] == address.lower():
                    count += 1
                    if res is None:
                        res = int(r['value'])
                    else:
                        res = min(res, int(r['value']))
                    if count == 10000:
                        return res/ETHER_VALUE
            page += 1
            resp = get_normal_transactions_address(address, page=page)
    except:
        return -10**22
    if res == None:
        return 0
    return res/ETHER_VALUE


def get_max_ether_sent_by_address(address):
    page = 1
    resp = get_normal_transactions_address(address, page=page)
    res = None
    count = 0
    try:
        while resp is not None and resp['status'] == '1':
            for r in resp['result']:
                if r['from'] == address.lower():
                    count += 1
                    if res is None:
                        res = int(r['value'])
                    else:
                        res = max(res, int(r['value']))
                    if count == 10000:
                        return res/ETHER_VALUE
            page += 1
            resp = get_normal_transactions_address(address, page=page)
    except:
        return -10**20
    if res == None:
        return 0
    return res/ETHER_VALUE


def get_average_ether_sent_by_address(address):
    page = 1
    resp = get_normal_transactions_address(address, page=page)
    total = 0
    count = 0
    while resp is not None and resp['status'] == '1':
        for r in resp['result']:
            #print(r['hash'])
            if r['from'] == address.lower():
                total += int(r['value'])
                count += 1
            if count == 10000:
                return (total/count)/ETHER_VALUE
        page += 1
        resp = get_normal_transactions_address(address, page=page)
    if count == 0:
        return 0
    return (total/count)/ETHER_VALUE


# print(get_balance_single_address("0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"))
# print(get_balance_multi_address(["0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"]))
address = "0xa8AeFBf7b044660cf0DFda1aEC1d30f68810e9C5"
to = "0xa8aefbf7b044660cf0dfda1aec1d30f68810e9c5"
# r = get_normal_transactions_address("0xc5102fE9359FD9a28f877a67E36B0F050d81a3CC", page=3)
# r = get_total_ether_received_by_address(address)
# mi = get_min_ether_received_by_address(address)
# ma = get_max_ether_received_by_address(address)
# av = get_average_ether_received_by_address(address)
# print(r)
# print(mi)
# print(ma)
# print(av)
# r = get_total_ether_sent_by_address(address)
# mi = get_min_ether_sent_by_address(address)
#ma = get_max_ether_sent_by_address(address)
# av = get_average_ether_sent_by_address(address)
# print(r)
# print(mi)
#print(ma)
# print(av)


#addresses = df["Address"]
API_KEY = "SVFQMQGPZ6NGU7GD5E281BGCZX1RTKU4Y5"

base_url = "https://api.etherscan.io/api"
def make_api_url(module, action, address, **kwargs):
	url = base_url + f"?module={module}&action={action}&address={address}&apikey={API_KEY}"

	for key, value in kwargs.items():
		url += f"&{key}={value}"

	return url




def get_time_diff(address):


    transactions_url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]

    internal_tx_url = make_api_url("account", "txlistinternal", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response2 = requests.get(internal_tx_url)
    data2 = response2.json()["result"]

    data.extend(data2)
    data.sort(key=lambda x: int(x['timeStamp']))


    if len(data) == 0:
        d = "NA"
    else:
        time_diff_unix = int(data[-1]["timeStamp"]) - int(data[0]["timeStamp"])

        time_diff = time_diff_unix/60
        d = time_diff
    
    return d

def get_mean_time_btw_received(address):
   

    transactions_url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]

    internal_tx_url = make_api_url("account", "txlistinternal", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
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
            
    
    return d
                                                                                                                                                                                                                                                                                                          

#print(get_mean_time_btw_received(addresses))
  
def get_mean_time_btw_sent(address):
    

    transactions_url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]

    internal_tx_url = make_api_url("account", "txlistinternal", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
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
            
    
    return d


#print(get_mean_time_btw_sent(addresses))


def get_num_received_sent(addresses):

  received = {}
  sent = {}

  for address in addresses[:10]:
    received_list = []
    sent_list =[]
    transactions_url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]
      
    internal_tx_url = make_api_url("account", "txlistinternal", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response2 = requests.get(internal_tx_url)
    data2 = response2.json()["result"]
      
    data.extend(data2)
    data.sort(key=lambda x: int(x['timeStamp']))
    

    for txn in data:
      if txn["from"] == address:
        sent_list.append(txn["to"])
      else:
        received_list.append(txn["from"])
    
    received[address] = len(set(received_list))
    sent[address] = len(set(sent_list))

  return received,sent



def get_total_erc20_txns(address):
    transaction_url = make_api_url("account", "tokentx", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transaction_url)
    data = response.json()["result"]
    return len(data)

#print(addresses[10])
#print(get_total_erc20_txns(addresses[10]))

def get_erc20_received_sent(address):
    received_list = []
    sent_list = []
    transactions_url = make_api_url("account", "tokentx", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]

    for txn in data:
        if txn["from"] == address:
            sent_list.append(txn["to"])
        else:
            received_list.append(txn["from"])

    received = list(set(received_list))
    sent = list(set(sent_list))

    return received,sent

# print(addresses[10])
# print(get_erc20_received_sent(addresses[10]))

def get_erc20_ether_sent(address):
    transactions_url = make_api_url("account", "tokentx", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]
    total = 0

    for txn in data:
        if txn['from'] == address.lower():
            total += int(txn['value'])

    return (total / ETHER_VALUE if total !=0 else 0)

# print(addresses[10])
# print(get_erc20_ether_sent(addresses[10]))#

def get_erc20_ether_received(address):
    transactions_url = make_api_url("account", "tokentx", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]
    total = 0

    for txn in data:
        if txn['to'] == address.lower():
            total += int(txn['value'])
    
    return (total / ETHER_VALUE if total !=0 else 0)

# print(addresses[10])
# print(get_erc20_ether_received(addresses[10]))


def get_erc20_received_sent(address):
    received_dict = {}
    sent_dict = {} 
    transactions_url = make_api_url("account", "tokentx", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]
    if (data):
        for txn in data:
            if (txn):
                if txn["from"] == address:
                    if txn["to"] not in sent_dict:
                        sent_dict[txn["to"]] = 1
                else:
                    if txn["from"] not in received_dict:
                        received_dict[txn["from"]] = 1 
    return len(received_dict), len(sent_dict)


#print(addresses[10])
#print(get_erc20_received_sent(addresses[10]))




def ERC20UniqueReceivedFrom_Addresses(address):
    received_dict = {}
    transactions_url = make_api_url("account", "tokentx", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]
    if (data):
        for txn in data:
            if (txn):
                if txn["to"] == address:
                    if txn["from"] not in received_dict:
                        received_dict[txn["from"]] = 1 
    return len(received_dict)


def ERC20UniqueSentFrom_Addresses(address):
    sent_dict = {} 
    transactions_url = make_api_url("account", "tokentx", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]
    if (data):
        for txn in data:
            if (txn):
                if txn["from"] == address:
                    if txn["to"] not in sent_dict:
                        sent_dict[txn["to"]] = 1
    return len(sent_dict)

#print(addresses[10])
#print(UniqueReceivedFrom_Addresses("0x1056d8d9ebb0e0d8710a0e2a1852d4a09d56464a"))
#print(UniqueSentFrom_Addresses(addresses[10]))

def numUniqSent_Addr(address):
    transactions_url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]
    sent_dict = {} 

    if (data):
        for txn in data:
            if (txn):
                if txn["from"] == address:
                    if txn["to"] not in sent_dict:
                        sent_dict[txn["to"]] = 1
    return len(sent_dict)



def numUniqReceived_Addr(address):
    transactions_url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    response = requests.get(transactions_url)
    data = response.json()["result"]
    received_dict = {} 
    if (data):
        for txn in data:
            if (txn):
                if txn["to"] == address:
                    if txn["from"] not in received_dict:
                        received_dict[txn["from"]] = 1 
    return len(received_dict)

# Get total transactions, sent transactions, received transactions of a wallet.
# Rationale of -(10**7) is that if we have an ERROR, the number of transactions will be < 0, indicating an ERROR of this function
def get_txn_statistics(addr, startblock = 0):
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
            elif (type(txn_info)) == list:
                txn_count = len(txn_info)
                from_lst = [x['from'].lower() for x in txn_info]
                sent_txn_count = from_lst.count(addr)
                to_lst = [x['to'].lower() for x in txn_info]
                received_txn_count = to_lst.count(addr)
            else:
                txn_count = -(10**7)
                sent_txn_count = -(10**7)
                received_txn_count = -(10**7)
        except:
            txn_count = -(10**7)
            sent_txn_count = -(10**7)
            received_txn_count = -(10**7)  
            print(f'Error in respsonse of API call for address: {addr}')          
    return txn_count, sent_txn_count, received_txn_count


# Get lime output for feature importance
# import lime
# import lime.lime_tabular
# explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train), feature_names=list(x_train.columns), discretize_continuous=True)
# i = 3 #testing on test data on row 3
# exp = explainer.explain_instance(np.array(x_test)[i], hgb.predict_proba, num_features=10) #using HistGradientBoosting model
# exp.show_in_notebook(show_table=True)

'''
#update dataset
df = pd.read_csv("fraud_n_non_fraud.csv")
df['Total Ether Sent'] = df["Address"].apply(get_total_ether_sent_by_address)
df['Max Ether Sent'] = df["Address"].apply(get_max_ether_sent_by_address)
df['Min Ether Sent'] = df["Address"].apply(get_min_ether_sent_by_address)
df['Average Ether Sent'] = df["Address"].apply(get_average_ether_sent_by_address)

#alternative method
df['Total Ether Sent'] = np.nan
for i in range(0, len(df)):
    df.loc[i, "Total Ether Sent"] = get_total_ether_sent_by_address(df.loc[i, "Address"])

df.to_csv("updated_austin.csv", index=False)
'''
