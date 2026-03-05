import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

import 'bert_wordpiece_tokenizer.dart';

/// 文字列のベクトル化に必要な処理をまとめたサービス。
/// 役割:
/// 1. ONNX Runtime の初期化/解放
/// 2. モデルと vocab のロード
/// 3. 文字列を token ids 化して推論実行
/// 4. last_hidden_state から文埋め込みを生成
class VectorizerService {
  VectorizerService({
    this.modelAssetPath = 'assets/models/BAAI_bge-small-en-v1.5.onnx',
    this.vocabAssetPath = 'assets/models/vocab.txt',
    this.maxLength = 128,
  });

  final String modelAssetPath;
  final String vocabAssetPath;
  final int maxLength;

  OrtSession? _session;
  BertWordPieceTokenizer? _tokenizer;

  Future<void> initialize() async {
    await _closeSession();

    // flutter_onnxruntime の createSession は通常のファイルパスを受け取る。
    // Flutter asset はそのままネイティブ側へ渡せないため、いったん実ファイル化する。
    // createSessionFromAsset も使えるが、内部キャッシュで古いモデルが残りうるので、
    // asset を差し替えた際に確実に反映されるよう毎回上書きしている。
    final modelPath = await _writeModelAssetToTempFile();

    // 語彙辞書をアセットから読み込む。
    final vocabText = await rootBundle.loadString(vocabAssetPath);

    // tokenizer / session を組み立て、以降の推論で再利用する。
    _tokenizer = BertWordPieceTokenizer.fromVocabText(vocabText);
    _session = await OnnxRuntime().createSession(modelPath);
  }

  Future<List<double>> embed(String text) async {
    final session = _session;
    final tokenizer = _tokenizer;
    if (session == null || tokenizer == null) {
      throw StateError('VectorizerService is not initialized.');
    }
    if (text.trim().isEmpty) {
      throw ArgumentError('Input text is empty.');
    }

    // 入力文を固定長 (maxLength) の input_ids / attention_mask に変換。
    final encoded = tokenizer.encode(text, maxLength: maxLength);
    final inputTensor = await OrtValue.fromList(
      Int64List.fromList(encoded.inputIds),
      [1, maxLength],
    );
    final attentionTensor = await OrtValue.fromList(
      Int64List.fromList(encoded.attentionMask),
      [1, maxLength],
    );

    Map<String, OrtValue>? outputs;
    try {
      // ONNX 推論実行。
      outputs = await session.run({
        'input_ids': inputTensor,
        'attention_mask': attentionTensor,
      });
      final outputValue = outputs['last_hidden_state'];
      if (outputValue == null) {
        throw StateError('Model output "last_hidden_state" is missing.');
      }
      final tensorData = await outputValue.asList();

      // last_hidden_state を attention_mask 付き mean pooling して文ベクトル化。
      return _meanPoolEmbedding(tensorData, encoded.attentionMask);
    } finally {
      // 推論ごとに生成した native tensor を確実に解放する。
      if (outputs != null) {
        for (final output in outputs.values) {
          await output.dispose();
        }
      }
      await inputTensor.dispose();
      await attentionTensor.dispose();
    }
  }

  void dispose() {
    unawaited(_closeSession());
  }

  Future<void> _closeSession() async {
    final session = _session;
    _session = null;
    if (session != null) {
      await session.close();
    }
  }

  Future<String> _writeModelAssetToTempFile() async {
    final data = await rootBundle.load(modelAssetPath);
    final directory = await getTemporaryDirectory();
    final fileName = modelAssetPath.split('/').last;
    final file = File('${directory.path}${Platform.pathSeparator}$fileName');
    await file.parent.create(recursive: true);
    await file.writeAsBytes(
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      flush: true,
    );
    return file.path;
  }

  static List<double> _meanPoolEmbedding(
    dynamic tensorValue,
    List<int> attentionMask,
  ) {
    // ONNX 出力から token ごとの hidden vectors を抽出。
    final tokenVectors = _extractTokenVectors(tensorValue);
    if (tokenVectors.isEmpty) {
      throw StateError('Model output is empty.');
    }
    if (attentionMask.isEmpty) {
      throw StateError('Attention mask is empty.');
    }

    // mask=1 の token のみ平均対象にする（PAD を除外）。
    final count = tokenVectors.length < attentionMask.length
        ? tokenVectors.length
        : attentionMask.length;
    final hiddenSize = tokenVectors.first.length;
    final sums = List<double>.filled(hiddenSize, 0);
    var validTokenCount = 0;
    for (var tokenIndex = 0; tokenIndex < count; tokenIndex++) {
      if (attentionMask[tokenIndex] == 0) {
        continue;
      }
      validTokenCount++;
      final tokenVector = tokenVectors[tokenIndex];
      for (var i = 0; i < hiddenSize; i++) {
        sums[i] += tokenVector[i];
      }
    }
    if (validTokenCount == 0) {
      throw StateError('No valid tokens for pooling.');
    }
    return sums.map((value) => value / validTokenCount).toList();
  }

  static List<List<double>> _extractTokenVectors(dynamic value) {
    // 想定は [batch, seq, hidden]。この実装では batch=1 前提で取り出す。
    if (value is! List || value.isEmpty) {
      throw StateError('Unexpected output format: $value');
    }

    final first = value.first;
    if (first is List && first.isNotEmpty && first.first is List) {
      return first
          .map(
            (row) => (row as List)
                .map((item) => (item as num).toDouble())
                .toList(growable: false),
          )
          .toList(growable: false);
    }

    if (first is List) {
      return value
          .map(
            (row) => (row as List)
                .map((item) => (item as num).toDouble())
                .toList(growable: false),
          )
          .toList(growable: false);
    }

    throw StateError('Unsupported tensor rank or type: $value');
  }
}
