# vectorize

Flutter + ONNX Runtime で、オフラインの意味検索を行うサンプルです。  
`BAAI/bge-small-en-v1.5` を使ってクエリ文を埋め込みベクトルへ変換し、事前計算済みの検索インデックスに対してコサイン類似度で上位結果を返します。

## 現在の実装

- `lib/main.dart`
  - 入力 UI（`Search word` / `Search` ボタン）
  - 起動時に ONNX Runtime と検索インデックスを初期化
  - 類似度上位 10 件を表示
- `lib/vectorizer_service.dart`
  - `assets/models/BAAI_bge-small-en-v1.5.onnx` と `assets/models/vocab.txt` を読み込み
  - `flutter_onnxruntime` で ONNX セッションを初期化
  - asset の ONNX モデルを一時ディレクトリへ書き出して `createSession()` に渡す
  - `input_ids` / `attention_mask` を `tensor(int64)` でモデルへ入力
  - `last_hidden_state` を attention mask 付き mean pooling して文ベクトル化
- `lib/bert_wordpiece_tokenizer.dart`
  - `vocab.txt` からトークン辞書を構築
  - lowercase + basic tokenization + WordPiece（`##` 継続トークン）で ID 化
  - `[CLS] ... [SEP]` 付与、`maxLength=128` へパディング
- `lib/search_index.dart`
  - `assets/data/search_index.json` を読み込み
  - 事前計算済み埋め込みインデックスをメモリ展開
- `lib/cosine_search.dart`
  - クエリ埋め込みを L2 正規化
  - 正規化済みインデックスとの内積でランキング
- `tools/main.py`
  - HuggingFace モデルを ONNX にエクスポート
  - Flutter 側互換のため IR version を調整（既定: `10`）
  - 単一 `.onnx` ファイル保存（external data 無効）
  - ORT で推論テストして最低限の健全性を確認
  - 任意で INT8 量子化
- `tools/build_index.py`
  - カタログ JSON から埋め込みを事前計算
  - L2 正規化済み `search_index.json` を生成

## Flutter 実行

```bash
fvm flutter pub get
fvm flutter run
```

`fvm` を使わない場合は、そのまま `flutter pub get` / `flutter run` に置き換えてください。

起動後にテキストを入力して `Search` を押すと、以下を確認できます。

- 類似度上位 10 件
- 各候補の score / title / genre / summary

## 検索インデックス生成

1. カタログを用意します。
2. Python ツールで埋め込みを事前計算します。

```bash
cd tools
uv sync
uv run build_index.py \
  --catalog ../assets/data/catalog.dummy.500.json \
  --onnx-model ../assets/models/BAAI_bge-small-en-v1.5.onnx \
  --tokenizer-model BAAI/bge-small-en-v1.5 \
  --output ../assets/data/search_index.json
```

生成される `search_index.json` は概ね以下の形式です。

```json
{
  "model": "BAAI_bge-small-en-v1.5.onnx",
  "dimension": 384,
  "items": [
    {
      "id": "book-001",
      "title": "...",
      "summary": "...",
      "genre": "...",
      "embedding": [0.0123, -0.0045, "..."]
    }
  ]
}
```

## モデル再生成

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
uv run main.py --model BAAI/bge-small-en-v1.5 --output ../assets/models --target-ir-version 10
```

INT8 量子化する場合:

```bash
uv run main.py --model BAAI/bge-small-en-v1.5 --output ../assets/models --quantize
```

## 既定アセット

- `assets/models/BAAI_bge-small-en-v1.5.onnx`
- `assets/models/vocab.txt`
- `assets/data/search_index.json`
- `assets/data/catalog.sample.json`
- `assets/data/catalog.dummy.500.json`

## トラブルシュート

- `Unsupported model IR version` が出る場合:
  - `tools/main.py` で `--target-ir-version 10` を指定して再生成してください。
- `PlatformException(INFERENCE_ERROR, Unexpected input data type...)` が出る場合:
  - モデルが `tensor(int64)` を期待しているので、アプリ側が最新の `Int64List` 入力実装になっているか確認してください。
- 初期化失敗 (`Initialization failed`) の場合:
  - `pubspec.yaml` の asset パスとファイル実体が一致しているか確認してください。
- 検索結果が 0 件の場合:
  - `assets/data/search_index.json` に `items` が入っているか確認してください。
