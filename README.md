# vectorize

Flutter + ONNX Runtime でテキスト埋め込みをオンデバイス生成するサンプルです。  
現在は `BAAI/bge-small-en-v1.5` を対象に、入力文を WordPiece でトークナイズして推論し、mean pooling した文ベクトルを表示します。

## 現在の実装

- `lib/main.dart`
  - 入力 UI（`Input text` / `Vectorize` ボタン）
  - 初期化状態・推論状態の表示
  - 推論結果として token 列と埋め込み次元、先頭 16 要素を表示
- `lib/vectorizer_service.dart`
  - `assets/models/BAAI_bge-small-en-v1.5.onnx` と `assets/models/vocab.txt` を読み込み
  - ONNX Runtime セッション初期化（`OrtSession.fromBuffer`）
  - `input_ids` / `attention_mask` をモデルに入力して推論
  - `last_hidden_state` を attention mask 付き mean pooling して文ベクトル化
- `lib/bert_wordpiece_tokenizer.dart`
  - `vocab.txt` からトークン辞書を構築
  - lowercase + basic tokenization + WordPiece（`##` 継続トークン）で ID 化
  - `[CLS] ... [SEP]` 付与、`maxLength=128` へパディング
- `tools/main.py`
  - HuggingFace モデルを ONNX にエクスポート
  - Flutter 側互換のため IR version 調整（既定: `9`）
  - 単一 `.onnx` ファイル保存（external data 無効）
  - 任意で INT8 量子化

## Flutter 実行

```bash
fvm flutter pub get
fvm flutter run
```

> fvm を使用しない場合は `fvm` を省いて実行してください。
> ```bash
> flutter pub get
> flutter run
> ```

起動後にテキストを入力して `Vectorize` を押すと、以下を確認できます。

- 実際にモデルへ渡した token（PAD 除外）
- 埋め込みベクトル次元
- 埋め込み先頭 16 要素

## モデル再生成（Python / uv）

前提:

- Python `3.13+`
- `uv` インストール済み

```bash
cd tools
uv sync
uv run main.py --model BAAI/bge-small-en-v1.5 --output ../assets/models
```

IR version を明示する場合:

```bash
uv run main.py --model BAAI/bge-small-en-v1.5 --output ../assets/models --target-ir-version 9
```

INT8 量子化する場合:

```bash
uv run main.py --model BAAI/bge-small-en-v1.5 --output ../assets/models --quantize
```

## 既定アセット

- `assets/models/BAAI_bge-small-en-v1.5.onnx`
- `assets/models/vocab.txt`

## トラブルシュート

- `Unsupported model IR version` が出る場合:
  - `tools/main.py` で `--target-ir-version 9` を指定して再生成してください。
- 初期化失敗 (`Init failed`) の場合:
  - `pubspec.yaml` の asset パスとファイル実体が一致しているか確認してください。
