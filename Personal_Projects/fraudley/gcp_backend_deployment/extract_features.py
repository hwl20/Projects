from etherscan2 import *
import os
# from xml.dom.expatbuilder import FragmentBuilder
# import requests
import numpy as np
import pandas as pd 
from datetime import datetime

ETHER_VALUE = 10 ** 18
API_KEY = "SVFQMQGPZ6NGU7GD5E281BGCZX1RTKU4Y5"
api_key = API_KEY

def get_features(address):
    txn_statistics = get_txn_statistics(address)
    ERC20Unique = get_erc20_received_sent(address)
    minmax_ether_sentReceived = get_minmax_ether_sentReceived_by_address(address)
    ether_sent = get_total_average_ether_sent_by_address(address)
    ether_received = get_total_average_ether_received_by_address(address)
    d = {
        "numerical_balance": balance_checker(address),
        "txn_count": txn_statistics[0],
        "sent_txn" : txn_statistics[1],
        "received_txn": txn_statistics[2],
        "total_ether_sent": ether_sent[0],
        "max_ether_sent": minmax_ether_sentReceived[0],
        "min_ether_sent": minmax_ether_sentReceived[1],
        "average_ether_sent": ether_sent[1],
        "max_ether_received": minmax_ether_sentReceived[2],
        "min_ether_received": minmax_ether_sentReceived[3],
        "average_ether_received": ether_received[1],
        "total_ether_received": ether_received[0],
        "unique_received_from": txn_statistics[3],
        "unique_sent_to": txn_statistics[4],
        "get_time_diff": get_time_diff(address),
        "mean_time_btw_received": get_mean_time_btw_received(address),
        "mean_time_btw_sent": get_mean_time_btw_sent(address),
        "total_erc20_txns": get_total_erc20_txns(address),
        # "erc20_total_ether_sent": float(get_erc20_ether_sent(address)),
        # "erc20_total_ether_received": float(get_erc20_ether_received(address)),
        "erc20_total_ether_sent": ether_sent[0],
        "erc20_total_ether_received": ether_received[0],
        "erc20_unique_rec_add": ERC20Unique[0],
        "erc20_unique_sent_add": ERC20Unique[1]
    }
    print(d)
    df = pd.DataFrame.from_dict([d])
    
    
    return df

#print(get_features("0x26a40e8dbdb0dee17d7036fcc0a2ae3fecf4800d"))