import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("CLIENT_ID")
os.getenv("CLIENT_SECRET")
os.getenv("SCOPE")

def get_access_token():
    url = "https://idcs-83f94bb2f43a40919cf7722a3f011925.identity.oraclecloud.com/oauth2/v1/token"

    payload = f'grant_type=client_credentials&scope={os.getenv("SCOPE")}&client_id={os.getenv("CLIENT_ID")}&client_secret={os.getenv("CLIENT_SECRET")}'
    headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    d = response.json() 
    token = "Bearer " + d['access_token']
    return token