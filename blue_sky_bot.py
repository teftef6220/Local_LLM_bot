from config.all_config import get_all_args

import logging
import os
import torch
import sys
import re
import time
import typing as t
from datetime import datetime, timedelta, timezone

from llm_utils.llm_utiils import Language_model

# import openai
from atproto import Client ,models
from dateutil.parser import parse
from dotenv import load_dotenv

load_dotenv(verbose=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


load_dotenv()
HANDLE =os.environ["BS_USER_NAME"]
PASSWORD = os.environ["BS_PASSWORD"]


class LLMMessage(t.TypedDict):
    role: str
    content: t.Optional[str]
    name: t.Optional[str]
    function_call: t.Optional[t.Dict]


def get_notifications(client: Client):
    response = client.app.bsky.notification.list_notifications()
    return response.notifications


def update_seen(client: Client, seenAt: datetime):
    response = client.app.bsky.notification.update_seen({"seenAt": seenAt.isoformat()})
    return


def filter_mentions_and_replies_from_notifications(ns: t.List["models.AppBskyNotificationListNotifications.Notification"]) -> t.List[models.AppBskyNotificationListNotifications.Notification]:
    return [n for n in ns if n.reason in ("mention", "reply")]


def filter_unread_notifications(ns: t.List["models.AppBskyNotificationListNotifications.Notification"], seen_at: datetime) -> t.List["models.AppBskyNotificationListNotifications.Notification"]:
    # IndexされてからNotificationで取得できるまでにラグがあるので、最後に見た時刻より少し前ににIndexされたものから取得する
    return [n for n in ns if seen_at - timedelta(minutes=2) < parse(n.indexed_at)]


def get_thread(client: Client, uri: str) -> "models.AppBskyFeedDefs.FeedViewPost":
    return client.app.bsky.feed.get_post_thread({"uri": uri})


# TODO: receive models.AppBskyFeedDefs.ThreadViewPost
def is_already_replied_to(feed_view: models.AppBskyFeedDefs.FeedViewPost, did: str) -> bool:
    replies = feed_view.thread.replies
    if replies is None:
        return False
    else:
        return any([reply.post.author.did == did for reply in replies])


def flatten_posts(thread: "models.AppBskyFeedDefs.ThreadViewPost") -> t.List[t.Dict[str, any]]:
    posts = [thread.post]

    parent = thread.parent
    if parent is not None:
        posts.extend(flatten_posts(parent))

    return posts


def get_openai_chat_message_name(name: str) -> str:
    # should be '^[a-zA-Z0-9_-]{1,64}$'
    return name.replace(".", "_")


def posts_to_sorted_messages(posts: t.List[models.AppBskyFeedDefs.PostView], assistant_did: str) -> t.List[LLMMessage]:
    sorted_posts = sorted(posts, key=lambda post: post.indexed_at)
    messages = []
    for post in sorted_posts:
        role = "assistant" if post.author.did == assistant_did else "user"
        messages.append(LLMMessage(role=role, content=post.record.text, name=get_openai_chat_message_name(post.author.handle)))
    return messages


def thread_to_messages(thread: "models.AppBskyFeedGetPostThread.Response", did: str) -> t.List[LLMMessage]:
    if thread is None:
        return []
    posts = flatten_posts(thread.thread)
    messages = posts_to_sorted_messages(posts, did)
    return messages


def generate_reply(post_messages: t.List[LLMMessage]):
    # <https://platform.openai.com/docs/api-reference/chat/create>
    # messages = [{"role": "system", "content": "Reply friendly in 280 characters or less. No @mentions."}]
    # messages.extend(post_messages)
    # chat_completion = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=messages,
    # )
    # first = chat_completion.choices[0]
    # return first.message.content

    #[{'role': 'user', 'content': '@latextex.bsky.social\u3000あなたの名前は？', 'name': 'userneme_bsky_social'}]
    message_only = post_messages[0]['content']
    pattern_bsky = r".*bsky\.social\s*(.*)"
    match_bsky = re.search(pattern_bsky, message_only)

    result_mesage = match_bsky.group(1) if match_bsky else None

    if result_mesage is not None:

        final_prompt = f"""指示:\n{result_mesage}\n応答:"""

        print(final_prompt)

        input_ids = mafuyu_tokenizer.encode(final_prompt, add_special_tokens=False, return_tensors="pt")

        output_ids = mafuyu_model.generate(
            input_ids=input_ids.to(device=model.device),
            max_length=200,
            temperature=0.7,
            do_sample=True,
        )

        output = mafuyu_tokenizer.decode(output_ids.tolist()[0][input_ids.size(1):])

        split_latest_texts = re.split(r"(応答:)", output)  
        try:
            last_response_latest = split_latest_texts[-1].rstrip("</s>")  
        except IndexError:
            last_response_latest = "ごめん...エラーが出たみたい..."

    else:
        last_response_latest = "ごめん...よく聞こえなかった..."

    return last_response_latest 


def reply_to(notification: models.AppBskyNotificationListNotifications.Notification) -> t.Union[models.AppBskyFeedPost.ReplyRef, models.AppBskyFeedDefs.ReplyRef]:
    parent = {
        "cid": notification.cid,
        "uri": notification.uri,
    }
    if notification.record.reply is None:
        return {"root": parent, "parent": parent}
    else:
        return {"root": notification.record.reply.root, "parent": parent}


def read_notifications_and_reply(client: Client, last_seen_at: datetime = None) -> datetime:
    logging.info(f"last_seen_at: {last_seen_at}")
    did = client.me.did

    seen_at = datetime.now(tz=timezone.utc)

    # unread countで判断するアプローチは、たまたまbsky.appで既読をつけてしまった場合に弱い
    ns = get_notifications(client)
    ns = filter_mentions_and_replies_from_notifications(ns)
    print("-------------------",last_seen_at)
    if last_seen_at is not None:
        ns = filter_unread_notifications(ns, last_seen_at)

    if (len(ns) == 0):
        logging.info("No unread notifications")  # avoid to call update_seen unnecessarily.
        return seen_at

    for notification in ns:
        thread = get_thread(client, notification.uri)
        if is_already_replied_to(thread, did):
            logging.info(f"Already replied to {notification.uri}")
            continue

        post_messages = thread_to_messages(thread, did)
        reply = generate_reply(post_messages)
        client.send_post(text=f"{reply}", reply_to=reply_to(notification))

    update_seen(client, seen_at)
    return seen_at


def login(client: Client, initial_wait: int):
    sleep_duration = initial_wait
    max_sleep_duration = 3600  # 1 hour

    while True:
        try:
            client.login(HANDLE, PASSWORD)
            return  # if login is successful, exit the loop
        except Exception as e:
            logging.exception(f"An error occurred during login: {e}")
            if sleep_duration > max_sleep_duration:  # if sleep duration has reached the max, exit the system
                logging.error("Max sleep duration reached, exiting system.")
                sys.exit(1)
            time.sleep(sleep_duration)
            sleep_duration *= 2  # double the sleep duration on failure


def main():
    client = Client(base_url="https://bsky.social")
    login(client, initial_wait=1)
    seen_at = None
    print("start")
    while True:
        try:
            seen_at = read_notifications_and_reply(client, seen_at)
        except Exception as e:
            logging.exception(f"An error occurred: {e}")
            login(client, initial_wait=60)
        finally:
            time.sleep(10)


if __name__ == "__main__":

    args = get_all_args()

    model_dir = os.path.join(args.model_base_dir, args.model_instance_dir)
    
    model = Language_model(args, args.llm_model_name, model_dir, args.tokenizer_name, "cuda")

    mafuyu_model = model.prepare_models(quantization_type = "nf4",precision = torch.float16)

    mafuyu_tokenizer = model.prepare_tokenizer()

    main()