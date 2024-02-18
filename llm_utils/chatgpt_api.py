from openai import OpenAI
import os
from dotenv import load_dotenv

'''
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
        "role": "assistant"
      },
      "logprobs": null
    }
  ],
  "created": 1677664795,
  "id": "chatcmpl,
  "model": "gpt-3.5-turbo-0613",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 17,
    "prompt_tokens": 57,
    "total_tokens": 74
  }
}
'''

load_dotenv(verbose=True)


class ChatGPTAPI:
    def __init__(self):
        self.client = OpenAI()
        self.api_key = os.environ["OPENAI_KEY"]
        self.client.api_key = self.api_key

    def chat(self, prompt):
        response = self.client.chat.completions.create(
                    model = "gpt-3.5-turbo-16k-0613",
                    messages = [
                        {"role": "system", "content": "You name is Asahina Mafuyu."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
        return response.choices[0].message.content