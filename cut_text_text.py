# import re

# latest_text = " たいっていっただけ"

import re

# 対象の文字列
text = "@latextex.bsky.social\u3000あｇじぇいｐじょえｈｐｗ？"

# "bsky.social"を含め、それ以前の部分をすべて消去する正規表現パターンを適用
pattern_bsky = r".*bsky\.social\s*(.*)"

# 正規表現を使ってパターンに一致する部分を検索
match_bsky = re.search(pattern_bsky, text)

# 一致した部分があれば、そのグループを取り出す
result_bsky = match_bsky.group(1) if match_bsky else "一致する部分がありません"

result_bsky
print(result_bsky)
# split_latest_texts = re.split(r"(応答:)", latest_text)  # 「応答: 」で分割し、デリミタもリストに含める

# try:
#     last_response_latest = split_latest_texts[-1].rstrip("</s>")  # 最後の部分から「</s>」を取り除く
# except IndexError:
#     last_response_latest = "ごめん...エラーが出たみたい..."

# last_response_latest


# print(last_response_latest)