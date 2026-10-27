import unittest
from .collector import teacher_record, validate_sources
from .games import plans


class CollectionContractTest(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(train_seeds=[1,2],reserved_test_seeds=[3,4],opponents=['round2'],seats=[0,1])
        self.sources = [dict(game=p['game'],seed=p['seed'],side=p['side'],
            trace=dict(path='traces/'+p['game']+'.jsonl.gz',size=1,sha256='a'*64),
            replay_sha256='b'*64,audit_sha256='c'*64) for p in plans(self.cfg)]

    def test_teacher_must_be_the_actor_choice(self):
        teacher_record(dict(source='teacher',teacher_label='unit:1',chosen='unit:1'))
        for value in (dict(source='callback',teacher_label='unit:1',chosen='unit:1'),
                      dict(source='teacher',teacher_label='unit:1',chosen='unit:2'),
                      dict(source='teacher',chosen='unit:1')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                teacher_record(value)

    def test_manifest_cannot_swap_or_relabel_sources(self):
        validate_sources(self.sources,self.cfg)
        for altered in (self.sources[::-1],self.sources[:-1],
            [{**self.sources[0],'trace':{**self.sources[0]['trace'],'path':'traces/other.jsonl.gz'}},*self.sources[1:]],
            [{**self.sources[0],'seed':3},*self.sources[1:]]):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                validate_sources(altered,self.cfg)


if __name__ == '__main__':
    unittest.main()
