/// BERT 系モデル向けの最小 WordPiece tokenizer 実装。
/// 目的:
/// - アプリ内で `vocab.txt` から token/id 変換を行う
/// - ONNX モデルに渡す `input_ids` と `attention_mask` を生成する
class BertWordPieceTokenizer {
  BertWordPieceTokenizer._(
    this._tokenToId,
    this._idToToken,
    this.unkId,
    this.clsId,
    this.sepId,
    this.padId,
  );

  final Map<String, int> _tokenToId;
  final List<String> _idToToken;
  final int unkId;
  final int clsId;
  final int sepId;
  final int padId;

  factory BertWordPieceTokenizer.fromVocabText(String vocabText) {
    // vocab.txt は 1 行 1 token、行番号が token id に対応する形式。
    final lines = vocabText
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .toList(growable: false);

    final tokenToId = <String, int>{};
    for (var i = 0; i < lines.length; i++) {
      tokenToId[lines[i]] = i;
    }

    int findOrThrow(String token) {
      // BERT 推論に必須な特殊 token が欠けている場合は即エラーにする。
      final id = tokenToId[token];
      if (id == null) {
        throw StateError('Required token missing in vocab: $token');
      }
      return id;
    }

    return BertWordPieceTokenizer._(
      tokenToId,
      lines,
      findOrThrow('[UNK]'),
      findOrThrow('[CLS]'),
      findOrThrow('[SEP]'),
      findOrThrow('[PAD]'),
    );
  }

  ({List<int> inputIds, List<int> attentionMask}) encode(
    String text, {
    int maxLength = 128,
  }) {
    if (maxLength < 3) {
      throw ArgumentError('maxLength must be >= 3');
    }

    // BGE/BERT 前提で lowercase 正規化してから分割。
    final normalized = text.toLowerCase();
    final basicTokens = _basicTokenize(normalized);

    // basic token を WordPiece に細分化し、id 列を作る。
    final wordPieceIds = <int>[];
    for (final token in basicTokens) {
      wordPieceIds.addAll(_wordPieceTokenizeToIds(token));
    }

    // [CLS] ... [SEP] を付与してモデル入力形式に合わせる。
    final inputIds = <int>[clsId, ...wordPieceIds, sepId];
    // 長すぎる場合は末尾を切り詰め、最後を [SEP] に保つ。
    if (inputIds.length > maxLength) {
      inputIds
        ..removeRange(maxLength - 1, inputIds.length)
        ..[maxLength - 1] = sepId;
    }

    // 実 token は 1、PAD は 0 の attention mask を作る。
    final attentionMask = List<int>.filled(
      inputIds.length,
      1,
      growable: true,
    );
    // maxLength まで PAD で右詰めする。
    while (inputIds.length < maxLength) {
      inputIds.add(padId);
      attentionMask.add(0);
    }

    return (inputIds: inputIds, attentionMask: attentionMask);
  }

  List<String> decodeIds(List<int> ids) {
    // 可視化用途: id 列を token 文字列へ戻す（範囲外 id は無視）。
    return ids
        .where((id) => id >= 0 && id < _idToToken.length)
        .map((id) => _idToToken[id])
        .toList(growable: false);
  }

  List<String> _basicTokenize(String text) {
    // 英数字連続か、1 文字記号かで粗く分割する簡易 tokenizer。
    final matches = RegExp(r"[a-z0-9]+|[^\s\w]", unicode: true).allMatches(text);
    return matches
        .map((match) => match.group(0))
        .whereType<String>()
        .toList(growable: false);
  }

  List<int> _wordPieceTokenizeToIds(String token) {
    if (token.isEmpty) {
      return const [];
    }
    // token 全体が vocab にある場合はそのまま使用。
    final direct = _tokenToId[token];
    if (direct != null) {
      return [direct];
    }

    // 最長一致で [foo, ##bar, ...] に分割する。
    // どこでも分割不能なら [UNK] にフォールバック。
    final pieces = <int>[];
    var start = 0;
    while (start < token.length) {
      int? currentId;
      var end = token.length;
      while (start < end) {
        final sub = token.substring(start, end);
        final candidate = start == 0 ? sub : '##$sub';
        final id = _tokenToId[candidate];
        if (id != null) {
          currentId = id;
          break;
        }
        end--;
      }
      if (currentId == null) {
        return [unkId];
      }
      pieces.add(currentId);
      start = end;
    }
    return pieces;
  }
}
