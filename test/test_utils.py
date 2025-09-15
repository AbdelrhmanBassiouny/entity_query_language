from entity_query_language.cache_data import CacheDict, IndexedCache


def test_cache_insert():
    cache = IndexedCache([1, 2, 3])
    cache.insert({1: 'a', 2: 'a', 3: 'a'}, '1a2a3a')
    cache.insert({1: 'a', 2: 'a', 3: 'b'}, '1a2a3b')
    cache.insert({1: 'a', 2: 'b', 3: 'a'}, '1a2b3a')
    cache.insert({1: 'a', 2: 'b', 3: 'b'}, '1a2b3b')
    cache.insert({1: 'b', 2: 'a', 3: 'a'}, '1b2a3a')
    cache.insert({1: 'b', 2: 'a', 3: 'b'}, '1b2a3b')
    cache.insert({1: 'b', 2: 'b', 3: 'a'}, '1b2b3a')
    cache.insert({1: 'b', 2: 'b', 3: 'b'}, '1b2b3b')
    assert cache.cache == CacheDict({'a':
                                         CacheDict({'a': CacheDict({'a': '1a2a3a',
                                                                         'b': '1a2a3b'}),
                                                    'b': CacheDict({'a': '1a2b3a',
                                                                         'b': '1a2b3b'})}),
                                     'b':
                                         CacheDict({'a': CacheDict({'a': '1b2a3a',
                                                                         'b': '1b2a3b'}),
                                                    'b': CacheDict({'a': '1b2b3a',
                                                                         'b': '1b2b3b'})})
                                     })


def test_cache_retrieve():
    data = CacheDict({'a':
                          CacheDict({'a': CacheDict({'a': '1a2a3a',
                                                          'b': '1a2a3b'}),
                                     'b': CacheDict({'a': '1a2b3a',
                                                          'b': '1a2b3b'})}),
                      'b':
                          CacheDict({'a': CacheDict({'a': '1b2a3a',
                                                          'b': '1b2a3b'}),
                                     'b': CacheDict({'a': '1b2b3a',
                                                          'b': '1b2b3b'})})
                      })
    assignment = {2: 'a'}
    cache = IndexedCache([1, 2, 3])
    result = list(cache.retrieve(assignment, data))
    patters = [r[0] for r in result]
    output_vals = [r[1] for r in result]
    assert len(result) == 4, "Should generate 4 results"
    assert output_vals == ['1a2a3a', '1a2a3b', '1b2a3a', '1b2a3b']
    assert patters == [{1: 'a', 2: 'a', 3: 'a'}, {1: 'a', 2: 'a', 3: 'b'},
                       {1: 'b', 2: 'a', 3: 'a'}, {1: 'b', 2: 'a', 3: 'b'}]
