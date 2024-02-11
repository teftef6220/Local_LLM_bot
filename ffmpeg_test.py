import subprocess

def test_ffmpeg_with_wav(input_wav_file):
    # 変換後のファイル名
    output_wav_file = "output_test.wav"

    # ffmpegを使用してWAVファイルのサンプルレートを変更するコマンド
    command = [
        "ffmpeg",
        "-i", input_wav_file,  # 入力ファイル
        "-ar", "44100",  # サンプルレートを44100Hzに設定
        "-y",  # 出力ファイルが存在する場合は上書き
        output_wav_file  # 出力ファイル
    ]

    try:
        # コマンドを実行
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpegが正常に機能しました。")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("ffmpegでエラーが発生しました。")
        print(e.stderr.decode())

# WAVファイルのパスを指定
input_wav_file = "audio.wav"

# 関数を呼び出してffmpegのテストを実行
test_ffmpeg_with_wav(input_wav_file)