import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

import 'bert_wordpiece_tokenizer.dart';

/// UI 層へ返す推論結果。
/// - `embedding`: mean pooling 後の文ベクトル
/// - `tokens`: 実際にモデルへ入れた token 列（PAD を除く）
class VectorizationResult {
  VectorizationResult({
    required this.embedding,
    required this.tokens,
  });

  final List<double> embedding;
  final List<String> tokens;
}

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
  OrtSessionOptions? _sessionOptions;
  BertWordPieceTokenizer? _tokenizer;
  bool _ortInitialized = false;

  Future<void> initialize() async {
    // ONNX Runtime を初期化し、モデルセッションを構築する。
    OrtEnv.instance.init();
    _ortInitialized = true;
    _sessionOptions = OrtSessionOptions();

    // モデル本体と語彙辞書をアセットから読み込む。
    final modelBytes = (await rootBundle.load(modelAssetPath)).buffer.asUint8List();
    final vocabText = await rootBundle.loadString(vocabAssetPath);

    // tokenizer / session を組み立て、以降の推論で再利用する。
    _tokenizer = BertWordPieceTokenizer.fromVocabText(vocabText);
    _session = OrtSession.fromBuffer(modelBytes, _sessionOptions!);
  }

  Future<VectorizationResult> vectorize(String text) async {
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
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      encoded.inputIds,
      [1, maxLength],
    );
    final attentionTensor = OrtValueTensor.createTensorWithDataList(
      encoded.attentionMask,
      [1, maxLength],
    );
    final runOptions = OrtRunOptions();

    try {
      // ONNX 推論実行。
      final outputs = session.run(
        runOptions,
        {'input_ids': inputTensor, 'attention_mask': attentionTensor},
      );
      final firstTensor = outputs.firstWhere((value) => value is OrtValueTensor);
      // last_hidden_state を attention_mask 付き mean pooling して文ベクトル化。
      final embedding = _meanPoolEmbedding(
        (firstTensor as OrtValueTensor).value,
        encoded.attentionMask,
      );
      // 画面表示用 token からは PAD を除外する。
      final nonPadLength = encoded.attentionMask.where((m) => m == 1).length;
      final tokenIds = encoded.inputIds.take(nonPadLength).toList(growable: false);

      // onnxruntime が返した出力バッファを解放。
      for (final output in outputs) {
        output?.release();
      }

      return VectorizationResult(
        embedding: embedding,
        tokens: tokenizer.decodeIds(tokenIds),
      );
    } finally {
      // 入出力 tensor と run options は毎回作るため必ず解放する。
      runOptions.release();
      inputTensor.release();
      attentionTensor.release();
    }
  }

  void dispose() {
    // initialize 済みリソースを破棄する。
    _session?.release();
    _sessionOptions?.release();
    if (_ortInitialized) {
      OrtEnv.instance.release();
    }
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
