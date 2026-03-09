"""
ONNX Embedding Model Exporter
HuggingFace のテキスト埋め込みモデルを ONNX 形式に変換します。

Usage:
    uv run main.py [OPTIONS]

Options:
    --model TEXT     HuggingFace モデル名 (default: BAAI/bge-small-en-v1.5)
    --output PATH    出力先ディレクトリ (default: ../assets/models)
    --target-ir-version INT  出力 ONNX の IR バージョン (default: 10)
    --quantize       INT8 量子化を適用する
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer


def save_single_file_model(model: onnx.ModelProto, path: Path) -> None:
    # Flutter 側は `fromBuffer` で単一 .onnx を読む前提なので、
    # external data 参照を使わず 1 ファイルにまとめて保存する。
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=False,
        all_tensors_to_one_file=True,
    )
    # 以前の出力で .onnx.data が残っている可能性があるため掃除する。
    external_data_path = path.with_suffix(path.suffix + ".data")
    if external_data_path.exists():
        external_data_path.unlink()


def export_to_onnx(
    model_name: str,
    output_dir: Path,
    quantize: bool,
    target_ir_version: int,
) -> Path:
    # 1) HuggingFace から tokenizer / model をロード
    print(f"[1/4] モデルを読み込み中: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # torch.onnx.export 用のサンプル入力。
    # 入力 shape を固定できれば内容自体は任意でよい。
    dummy_text = "This is a sample sentence."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    onnx_path = output_dir / f"{safe_name}.onnx"

    # 2) ONNX エクスポート
    print(f"[2/4] ONNX にエクスポート中: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(inputs["input_ids"], inputs["attention_mask"]),
            f=str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
                "pooler_output": {0: "batch_size"},
            },
            opset_version=17,
        )

    # 3) ONNX 検証と互換性調整
    # - IR version を Flutter 側ランタイムに合わせる
    # - 単一ファイル形式で保存し直す
    print("[3/4] モデルを検証中...")
    onnx_model = onnx.load(str(onnx_path))
    if onnx_model.ir_version > target_ir_version:
        print(
            f"  IR version を {onnx_model.ir_version} -> {target_ir_version} に調整します"
        )
        onnx_model.ir_version = target_ir_version
    save_single_file_model(onnx_model, onnx_path)
    onnx.checker.check_model(onnx_model)

    if quantize:
        # オプション: 動的量子化でモデルサイズ削減。
        # 量子化後モデルも同じ互換性ルール(IR/単一ファイル)を適用する。
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = output_dir / f"{safe_name}_int8.onnx"
        print(f"  INT8 量子化中: {quantized_path}")
        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QInt8,
        )
        quantized_model = onnx.load(str(quantized_path))
        if quantized_model.ir_version > target_ir_version:
            print(
                f"  量子化後 IR version を {quantized_model.ir_version} -> {target_ir_version} に調整します"
            )
            quantized_model.ir_version = target_ir_version
        save_single_file_model(quantized_model, quantized_path)
        onnx.checker.check_model(quantized_model)
        onnx_path = quantized_path

    # 4) onnxruntime で実際に 1 回推論して最低限の健全性を確認
    print("[4/4] 推論テスト中...")
    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    outputs = session.run(None, ort_inputs)
    embedding = outputs[0][0].mean(axis=0)  # mean pooling
    print(f"  埋め込みベクトル次元: {embedding.shape[0]}")
    print(f"  L2ノルム: {np.linalg.norm(embedding):.4f}")

    print(f"\n完了: {onnx_path}")
    return onnx_path


def main() -> None:
    # CLI 引数を受け取り、変換処理を 1 回実行するエントリポイント。
    parser = argparse.ArgumentParser(
        description="HuggingFace モデルを ONNX 形式にエクスポートします"
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace モデル名 (default: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "assets" / "models"),
        help="出力先ディレクトリ (default: ../assets/models)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="INT8 量子化を適用する",
    )
    parser.add_argument(
        "--target-ir-version",
        type=int,
        default=10,
        help="出力 ONNX の IR バージョン (default: 10)",
    )
    args = parser.parse_args()

    export_to_onnx(
        model_name=args.model,
        output_dir=Path(args.output),
        quantize=args.quantize,
        target_ir_version=args.target_ir_version,
    )


if __name__ == "__main__":
    main()
