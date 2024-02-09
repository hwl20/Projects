from etherscan2 import *
import pandas as pd 
import concurrent.futures

# import os
# from xml.dom.expatbuilder import FragmentBuilder
# import requests
# import numpy as np
# from datetime import datetime

ETHER_VALUE = 10 ** 18
API_KEY = "SVFQMQGPZ6NGU7GD5E281BGCZX1RTKU4Y5"
api_key = API_KEY

def get_features(address):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processing_txn_statistics = executor.submit(get_txn_statistics, address)
        processing_ERC20Unique = executor.submit(get_erc20_received_sent, address)
        processing_minmax_ether_sentReceived = executor.submit(get_minmax_ether_sentReceived_by_address, address)
        processing_ether_sent = executor.submit(get_total_average_ether_sent_by_address, address)
        processing_ether_received = executor.submit(get_total_average_ether_received_by_address, address)

        processing_numerical_balance = executor.submit(balance_checker, address)
        processing_get_time_diff = executor.submit(get_time_diff, address)
        processing_mean_time_btw_received = executor.submit(get_mean_time_btw_received, address)
        processing_mean_time_btw_sent = executor.submit(get_mean_time_btw_sent, address)
        processing_total_erc20_txns = executor.submit(get_total_erc20_txns, address)

    # more than 1 ouptut
    R_txn_count, R_sent_txn, R_received_txn, R_unique_received_from, R_unique_sent_to = processing_txn_statistics.result()
    R_erc20_unique_rec_add, R_erc20_unique_sent_add = processing_ERC20Unique.result()
    R_max_ether_sent, R_min_ether_sent, R_max_ether_received, R_min_ether_received = processing_minmax_ether_sentReceived.result()
    R_total_ether_sent, R_average_ether_sent = processing_ether_sent.result()
    R_total_ether_received, R_average_ether_received = processing_ether_received.result()

    # 1 output
    R_numerical_balance = processing_numerical_balance.result()
    R_get_time_diff = processing_get_time_diff.result()
    R_mean_time_btw_received = processing_mean_time_btw_received.result()
    R_mean_time_btw_sent = processing_mean_time_btw_sent.result()
    R_total_erc20_txns = processing_total_erc20_txns.result()

    d = {
        "numerical_balance": R_numerical_balance,
        "txn_count": R_txn_count,
        "sent_txn" : R_sent_txn,
        "received_txn": R_received_txn,
        "total_ether_sent": R_total_ether_sent,
        "max_ether_sent": R_max_ether_sent,
        "min_ether_sent": R_min_ether_sent,
        "average_ether_sent": R_average_ether_sent,
        "max_ether_received": R_max_ether_received,
        "min_ether_received": R_min_ether_received,
        "average_ether_received": R_average_ether_received,
        "total_ether_received": R_total_ether_received,
        "unique_received_from": R_unique_received_from,
        "unique_sent_to": R_unique_sent_to,
        "get_time_diff": R_get_time_diff,
        "mean_time_btw_received": R_mean_time_btw_received,
        "mean_time_btw_sent": R_mean_time_btw_sent,
        "total_erc20_txns": R_total_erc20_txns,
        # "erc20_total_ether_sent": float(get_erc20_ether_sent(address)),
        # "erc20_total_ether_received": float(get_erc20_ether_received(address)),
        "erc20_total_ether_sent": R_total_ether_sent,
        "erc20_total_ether_received": R_total_ether_received,
        "erc20_unique_rec_add": R_erc20_unique_rec_add,
        "erc20_unique_sent_add": R_erc20_unique_sent_add
    }
    print(d)

    df = pd.DataFrame.from_dict([d])
    return df

#print(get_features("0x26a40e8dbdb0dee17d7036fcc0a2ae3fecf4800d"))