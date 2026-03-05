import 'dart:math' as math;

import 'search_index.dart';

class SearchMatch {
  SearchMatch({required this.item, required this.score});

  final SearchIndexItem item;
  final double score;
}

List<double> l2Normalize(List<double> values) {
  var squaredSum = 0.0;
  for (final value in values) {
    squaredSum += value * value;
  }
  if (squaredSum == 0) {
    return List<double>.filled(values.length, 0, growable: false);
  }
  final inverseNorm = 1.0 / math.sqrt(squaredSum);
  return values.map((value) => value * inverseNorm).toList(growable: false);
}

double dotProduct(List<double> left, List<double> right) {
  final count = left.length < right.length ? left.length : right.length;
  var result = 0.0;
  for (var i = 0; i < count; i++) {
    result += left[i] * right[i];
  }
  return result;
}

List<SearchMatch> topKCosineNormalized({
  required List<double> query,
  required SearchIndex index,
  int k = 10,
}) {
  final normalizedQuery = l2Normalize(query);
  final matches = <SearchMatch>[];
  for (final item in index.items) {
    if (item.embedding.length != normalizedQuery.length) {
      continue;
    }
    matches.add(
      SearchMatch(
        item: item,
        score: dotProduct(normalizedQuery, item.embedding),
      ),
    );
  }
  matches.sort((a, b) => b.score.compareTo(a.score));
  if (matches.length <= k) {
    return matches;
  }
  return matches.take(k).toList(growable: false);
}
