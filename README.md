# vectorize

Flutter で ONNX Runtime を使って埋め込みモデルを実行するサンプルです。

## 構成

- `lib/main.dart`: ONNX Runtime 初期化、モデル配置、サンプル推論
- `tools/main.py`: HuggingFace モデルを ONNX へ変換
- `assets/models/`: アプリ同梱する ONNX モデルファイル

## Flutter アプリ実行

```bash
./.fvm/flutter_sdk/bin/flutter pub get
./.fvm/flutter_sdk/bin/flutter run
```

アプリ起動後、テキストを入力して `Vectorize` を押すと、
入力文をトークナイズして推論し、埋め込みベクトルを表示します。
モデルはアセットを `rootBundle.load(...)` で読み込み、`OrtSession.fromBuffer(...)` で初期化します。

## モデル再生成 (Python)

```bash
cd tools
uv run main.py --model BAAI/bge-small-en-v1.5 --output ../assets/models
```

Flutter 側 ONNX Runtime 互換のため、既定で `IR version = 9` を使います。
また `fromBuffer` で読めるよう、モデルは単一 `.onnx` ファイルとして保存されます。
必要なら明示指定もできます:

```bash
uv run main.py --model BAAI/bge-small-en-v1.5 --output ../assets/models --target-ir-version 9
```

量子化する場合:

```bash
uv run main.py --model BAAI/bge-small-en-v1.5 --output ../assets/models --quantize
```

## 注意

- 現在の Flutter 側は「推論疎通確認」の最小実装です。
- トークナイザー連携は未実装で、入力は固定 token IDs を使用しています。
- `Unsupported model IR version` が出る場合は、`tools/main.py` でモデルを再生成してください。
