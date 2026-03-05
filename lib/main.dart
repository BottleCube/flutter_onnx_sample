import 'package:flutter/material.dart';

import 'cosine_search.dart';
import 'search_index.dart';
import 'vectorizer_service.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Semantic Search',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
      ),
      home: const SemanticSearchPage(),
    );
  }
}

class SemanticSearchPage extends StatefulWidget {
  const SemanticSearchPage({super.key});

  @override
  State<SemanticSearchPage> createState() => _SemanticSearchPageState();
}

class _SemanticSearchPageState extends State<SemanticSearchPage> {
  static const _searchIndexPath = 'assets/data/search_index.json';
  static const _maxResults = 10;

  final _inputController = TextEditingController();
  final _vectorizerService = VectorizerService();

  bool _isSearching = false;
  String _status = 'Initializing search engine...';
  SearchIndex? _searchIndex;
  List<SearchMatch> _matches = const [];

  @override
  void initState() {
    super.initState();
    _initializeApp();
  }

  Future<void> _initializeApp() async {
    try {
      await _vectorizerService.initialize();
      final index = await SearchIndex.loadFromAsset(_searchIndexPath);

      if (!mounted) {
        return;
      }
      setState(() {
        _searchIndex = index;
        _status = 'Ready (${index.items.length} items loaded)';
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _status = 'Initialization failed: $error';
      });
    }
  }

  Future<void> _search() async {
    if (_isSearching) {
      return;
    }

    final query = _inputController.text.trim();
    if (query.isEmpty) {
      setState(() {
        _status = 'Please enter text.';
        _matches = const [];
      });
      return;
    }

    setState(() {
      _isSearching = true;
      _status = 'Searching...';
      _matches = const [];
    });

    try {
      final index = _searchIndex;
      if (index == null) {
        throw StateError('Search index is not loaded.');
      }
      final embedding = await _vectorizerService.embed(query);
      final matches = topKCosineNormalized(
        query: embedding,
        index: index,
        k: _maxResults,
      );

      if (!mounted) {
        return;
      }
      setState(() {
        _status = 'Done (${matches.length} results)';
        _matches = matches;
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _status = 'Search failed: $error';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isSearching = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _inputController.dispose();
    _vectorizerService.dispose();
    super.dispose();
  }

  Widget _buildSearchResults() {
    if (_matches.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 16),
        const Text('Top 10 similar items'),
        const SizedBox(height: 8),
        ..._matches.asMap().entries.map((entry) {
          final rank = entry.key + 1;
          final match = entry.value;
          final item = match.item;
          return Card(
            child: ListTile(
              title: Text('$rank. ${item.title}'),
              subtitle: Text(
                'score=${match.score.toStringAsFixed(4)}\n'
                'genre: ${item.genre}\n'
                '${item.summary}',
              ),
              isThreeLine: true,
            ),
          );
        }),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Semantic Search')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Search word'),
            const SizedBox(height: 8),
            TextField(
              controller: _inputController,
              minLines: 2,
              maxLines: 4,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                hintText: 'Search words',
              ),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: _isSearching ? null : _search,
              child: const Text('Search'),
            ),
            const SizedBox(height: 12),
            Text(_status),
            _buildSearchResults(),
          ],
        ),
      ),
    );
  }
}
