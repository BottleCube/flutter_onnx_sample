// デバイス上でテキストを密なベクトル（埋め込み）に変換するアプリ。
// BAAI/bge-small-en-v1.5 モデルを ONNX Runtime でオンデバイス推論するため、
// ネットワーク接続なしで動作する。
// 生成された埋め込みは意味検索・類似度計算・分類などに利用できる。

import 'package:flutter/material.dart';

import 'vectorizer_service.dart';

void main() {
  // アプリの起点。画面ロジックは `VectorizeHomePage` に集約する。
  runApp(const MyApp());
}

/// アプリケーションのルートウィジェット。
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vectorize',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue)),
      home: const VectorizeHomePage(),
    );
  }
}

/// テキストベクトル化機能を提供するメイン画面。
/// ユーザーが入力した文章を埋め込みベクトルに変換し、
/// トークン列と先頭 16 次元の値を画面に表示する。
class VectorizeHomePage extends StatefulWidget {
  const VectorizeHomePage({super.key});

  @override
  State<VectorizeHomePage> createState() => _VectorizeHomePageState();
}

class _VectorizeHomePageState extends State<VectorizeHomePage> {
  // 入力テキストと推論処理のオーケストレーションを担当。
  // 実際の ONNX 推論ロジックは `VectorizerService` 側へ分離している。
  final _inputController =
      TextEditingController(text: 'This is a sample sentence.');
  final _vectorizerService = VectorizerService();

  bool _running = false;
  String _status = 'Initializing ONNX Runtime...';
  List<double>? _embedding;
  List<String>? _tokens;

  @override
  void initState() {
    super.initState();
    // 画面表示と同時に ONNX Runtime / モデルを初期化する。
    _initializeOrt();
  }

  Future<void> _initializeOrt() async {
    try {
      // モデル・vocab のロードと ONNX セッション作成。
      await _vectorizerService.initialize();

      if (!mounted) {
        return;
      }
      setState(() {
        _status = 'Ready';
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _status = 'Init failed: $error';
      });
    }
  }

  Future<void> _vectorizeInputText() async {
    // 連打による二重実行を抑止。
    if (_running) {
      return;
    }

    // 空入力は推論せずにガードする。
    final inputText = _inputController.text.trim();
    if (inputText.isEmpty) {
      setState(() {
        _status = 'Please enter text.';
        _embedding = null;
        _tokens = null;
      });
      return;
    }

    setState(() {
      _running = true;
      _status = 'Vectorizing...';
      _embedding = null;
    });

    try {
      // 文字列 -> token ids -> ONNX 推論 -> 埋め込み の一連をサービスに委譲。
      final result = await _vectorizerService.vectorize(inputText);

      if (!mounted) {
        return;
      }
      setState(() {
        _status = 'Done';
        _embedding = result.embedding;
        _tokens = result.tokens;
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _status = 'Inference failed: $error';
      });
    }
    if (mounted) {
      setState(() {
        _running = false;
      });
    }
  }

  @override
  void dispose() {
    // Controller / ネイティブリソースを明示解放する。
    _inputController.dispose();
    _vectorizerService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final embedding = _embedding;
    final tokens = _tokens;
    return Scaffold(
      appBar: AppBar(title: const Text('Vectorize')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Input text'),
            const SizedBox(height: 8),
            TextField(
              controller: _inputController,
              minLines: 2,
              maxLines: 4,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                hintText: 'Enter text to vectorize',
              ),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: _running ? null : _vectorizeInputText,
              child: const Text('Vectorize'),
            ),
            const SizedBox(height: 12),
            Text(_status),
            if (tokens != null) ...[
              const SizedBox(height: 12),
              Text('Tokens (${tokens.length}): ${tokens.join(" ")}'),
            ],
            if (embedding != null) ...[
              const SizedBox(height: 12),
              Text('Embedding dimension: ${embedding.length}'),
              const SizedBox(height: 8),
              Text(
                'First 16 values: ${embedding.take(16).map((v) => v.toStringAsFixed(4)).join(", ")}',
              ),
            ],
          ],
        ),
      ),
    );
  }
}
