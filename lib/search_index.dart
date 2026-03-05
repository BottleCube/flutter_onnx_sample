import 'dart:convert';

import 'package:flutter/services.dart';

class SearchIndexItem {
  SearchIndexItem({
    required this.id,
    required this.title,
    required this.summary,
    required this.genre,
    required this.embedding,
  });

  final String id;
  final String title;
  final String summary;
  final String genre;
  final List<double> embedding;

  factory SearchIndexItem.fromJson(Map<String, dynamic> json) {
    return SearchIndexItem(
      id: json['id'] as String,
      title: json['title'] as String,
      summary: (json['summary'] as String?) ?? '',
      genre: (json['genre'] as String?) ?? '',
      embedding: (json['embedding'] as List<dynamic>)
          .map((value) => (value as num).toDouble())
          .toList(growable: false),
    );
  }
}

class SearchIndex {
  SearchIndex({
    required this.model,
    required this.dimension,
    required this.items,
  });

  final String model;
  final int dimension;
  final List<SearchIndexItem> items;

  static Future<SearchIndex> loadFromAsset(String path) async {
    final jsonText = await rootBundle.loadString(path);
    final root = json.decode(jsonText) as Map<String, dynamic>;
    final dimension = root['dimension'] as int;
    final items = (root['items'] as List<dynamic>)
        .map((item) => SearchIndexItem.fromJson(item as Map<String, dynamic>))
        .where((item) => item.embedding.length == dimension)
        .toList(growable: false);

    return SearchIndex(
      model: root['model'] as String,
      dimension: dimension,
      items: items,
    );
  }
}
