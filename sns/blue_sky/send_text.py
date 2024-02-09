import base64
from io import BytesIO

import os
from atproto import Client, client_utils
from dotenv import load_dotenv
from PIL import Image


class Sns_settings():
    def __init__(self,sns_type):
        self.sns_type = sns_type


    def login_to_blusky(self):
        bluesky = Client(base_url="https://bsky.social")
        bluesky.login(login=os.environ["BS_USER_NAME"], password=os.environ["BS_PASSWORD"])

        return bluesky

