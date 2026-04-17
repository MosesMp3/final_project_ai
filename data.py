import requests, time
import os
from dotenv import load_dotenv


load_dotenv()

CLIENT_ID = os.environ["IGDB_CLIENT_ID"]
CLIENT_SECRET = os.environ["IGDB_CLIENT_SECRET"]


print(CLIENT_ID)
print(CLIENT_SECRET)
