# -*- coding: utf-8 -*-
"""
DeepFilterNetを使用して音声ファイルの背景ノイズを除去するスクリプト。

このスクリプトは、コマンドラインから指定された音声ファイルを読み込み、
DeepFilterNetの深層学習モデルを用いてスピーチエンハンスメント（ノイズ除去）
処理を行い、結果を新しいファイルとして保存します。

主な機能:
- コマンドラインからのファイルパス入力
- 処理後のファイルを「[元のファイル名]_denoise_DeepFilterNet.[拡張子]」として出力
- `--model` オプションによる使用モデルの切り替え機能 (DeepFilterNet, DeepFilterNet2, DeepFilterNet3)

使用法:
    python denoise_audio.py /path/to/your/audio.wav
    python denoise_audio.py --model DeepFilterNet3 /path/to/your/audio.wav
"""

import argparse
import sys
from pathlib import Path
import subprocess
import tempfile
import os

# PyTorchとDeepFilterNetのインポート。これらのライブラリが見つからない場合は
# ユーザーにインストールを促すメッセージを表示する。
try:
    import torch
    from df.enhance import enhance, init_df, load_audio, save_audio
    from df.utils import download_file
    import torchaudio
except ImportError:
    print("エラー: 必要なライブラリが見つかりません。")
    print("仮想環境を有効にして、以下のコマンドでインストールしてください:")
    print("pip install torch torchaudio")
    print("pip install deepfilternet")
    sys.exit(1)

def file_exists(path_str: str) -> Path:
    """
    argparseのtypeとして使用する関数。
    ファイルパスが存在するかを検証し、存在すればPathオブジェクトを返す。
    """
    path = Path(path_str)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"エラー: ファイル '{path_str}' が見つかりません。")
    return path

def resample_with_ffmpeg(input_path: Path, target_sr: int) -> Path | None:
    """
    ffmpegを使用してオーディオファイルをリサンプリングし、一時ファイルとして保存する。

    Args:
        input_path: 入力ファイルのパス。
        target_sr: 目標サンプリングレート。

    Returns:
        リサンプリングされた一時ファイルのPathオブジェクト。失敗した場合はNone。
    """
    try:
        # ffmpegが利用可能か確認
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("情報: ffmpegが見つからないか、正常に動作しません。", file=sys.stderr)
        return None

    print(f"情報: ffmpegを使用して{target_sr}Hzにリサンプリングしています...")
    # delete=Falseとすることで、close後もファイルが残るようにする
    with tempfile.NamedTemporaryFile(suffix=input_path.suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    
    command = [
        "ffmpeg",
        "-i", str(input_path),
        "-ar", str(target_sr),
        "-y",  # 常に上書き
        "-hide_banner",
        "-loglevel", "error",
        str(temp_path)
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"情報: 一時ファイルを作成しました: {temp_path}")
        return temp_path
    except subprocess.CalledProcessError as e:
        print(f"ffmpegの実行中にエラーが発生しました: {e.stderr}", file=sys.stderr)
        if temp_path.exists():
            os.remove(temp_path)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="DeepFilterNetを使用して音声ファイルのノイズを除去します。",
        formatter_class=argparse.RawTextHelpFormatter  # ヘルプメッセージの改行を保持
    )
    parser.add_argument(
        "input_path",
        type=file_exists,
        help="処理対象のノイズ付き音声ファイルへのパス。"
    )
    parser.add_argument(
        "-m", "--model",
        dest="model_name",
        default="DeepFilterNet2",
        choices=["DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3"],
        help=(
            "使用する事前学習済みモデルを選択します。\n"
            "  - DeepFilterNet:  v1, バランス型\n"
            "  - DeepFilterNet2: v2, 高速・軽量（デフォルト）\n"
            "  - DeepFilterNet3: v3, 最高音質追求型"
        )
    )

    args = parser.parse_args()

    # 2. パスとファイル名の準備
    input_path = args.input_path
    # pathlibを使用して、要求された仕様通りに出力ファイルパスを生成
    output_path = input_path.parent / f"{input_path.stem}_denoise_DeepFilterNet{input_path.suffix}"

    print(f"処理を開始します...")
    print(f"  入力ファイル: {input_path}")
    print(f"  出力ファイル: {output_path}")
    print(f"  使用モデル: {args.model_name}")

    temp_audio_path = None  # 一時ファイルのパスを保持する変数
    try:
        # 3. DeepFilterNetモデルの初期化
        # ユーザーが選択したモデルをロードする
        model, df_state, _ = init_df(args.model_name, post_filter=True)
        target_sr = df_state.sr()

        # 4. オーディオのサンプリングレートを確認し、必要であればリサンプリング
        try:
            audio_info = torchaudio.info(str(input_path))
            original_sr = audio_info.sample_rate
        except Exception:
            print("警告: 入力ファイルのサンプリングレートを事前に確認できませんでした。", file=sys.stderr)
            original_sr = -1  # 不明な場合は-1として、常にリサンプリングを試みる

        audio_to_load = input_path
        if original_sr != target_sr:
            print(
                f"情報: 入力ファイルのサンプリングレート ({original_sr} Hz) は "
                f"モデルの要求 ({target_sr} Hz) と異なります。"
            )
            resampled_path = resample_with_ffmpeg(input_path, target_sr)
            if resampled_path:
                audio_to_load = resampled_path
                temp_audio_path = resampled_path  # 後で削除するためにパスを保存
            else:
                print("情報: ffmpegでのリサンプリングに失敗、またはスキップされました。", file=sys.stderr)
                print("情報: torchaudioの内部機能を使用してリサンプリングします。", file=sys.stderr)
        
        # 5. オーディオのロード
        # モデルが必要とするサンプリングレート（df_state.sr()）でオーディオをロード
        # ffmpegでのリサンプリングが失敗した場合も、この関数がリサンプリングを行う
        audio, _ = load_audio(audio_to_load, sr=df_state.sr())
        
        print("ノイズ除去処理を実行中...（ファイルサイズによって時間がかかる場合があります）")

        # 6. ノイズ除去の実行
        # メインの強調処理
        enhanced_audio = enhance(model, df_state, audio)
        
        print("処理が完了しました。ファイルを保存しています...")

        # 7. 処理済みオーディオの保存
        save_audio(output_path, enhanced_audio, sr=df_state.sr())
        
        print(f"正常に終了しました。出力ファイル: {output_path}")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}", file=sys.stderr)
        print("処理を中断しました。", file=sys.stderr)
        sys.exit(1)
    finally:
        # 8. 一時ファイルのクリーンアップ
        if temp_audio_path and temp_audio_path.exists():
            print(f"情報: 一時ファイルを削除します: {temp_audio_path}")
            os.remove(temp_audio_path)


if __name__ == "__main__":
    main()