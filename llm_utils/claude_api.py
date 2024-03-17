from anthropic import Anthropic
from dotenv import load_dotenv
import os


load_dotenv(verbose=True)

class ClaudeAPI:
    def __init__(self):
       
        self.api_key = os.environ["CLAUDE3_API_KEY"]
        self.client = Anthropic(api_key=self.api_key)

    def chat(self, prompt):
        response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            )

        return response.content[0].text