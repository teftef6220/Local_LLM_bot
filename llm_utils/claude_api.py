from anthropic import Anthropic
from dotenv import load_dotenv
import os

'''
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello!"
    }
  ],
  "model": "claude-2.1",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 6
  }
}
'''


load_dotenv(verbose=True)

class ClaudeAPI:
    def __init__(self):
       
        self.api_key = os.environ["CLAUDE3_API_KEY"]
        self.client = Anthropic(api_key=self.api_key)

    def chat(self, prompt):
        response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=300,
                system = "あなたは朝比奈まふゆです。宮益坂女子学園の高校３年生です。",
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            )

        return response.content[0].text