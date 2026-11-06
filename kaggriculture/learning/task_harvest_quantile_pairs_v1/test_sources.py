"""Pure archive and job-point wiring checks; no historical file reads."""
from copy import deepcopy
import unittest
from . import contract,sources


def witness():
    files=[dict(root=root,path=f'row-{i:03}.json',bytes=0,sha256='a'*64)
           for root,n in (('private',20),('stage',45)) for i in range(n)]
    files[0]['bytes']=contract.PAIR_BYTES
    return dict(schema='task-harvest-return-pairs-publication-witness-v1',
        registration_commit=contract.PAIR_REG_COMMIT,domain='task-harvest-return-pairs-v1',files=files,
        bytes_total=contract.PAIR_BYTES,commitment_sha256=contract.PAIR_COMMITMENT)


def documents():
    output=[]
    for source in contract.sources():
        points=[dict(label=q,index=i+1,target=dict(step=250+i,synthetic=q),reason=None)
                for i,q in enumerate(contract.POINTS)]
        output.append(dict(origin=source['role'],source=source,
            quantiles=dict(points=points,counts={'synthetic':True}),trace_index=[],trace_scans=[]))
    return output


class SourceContracts(unittest.TestCase):
    def test_inventory_binds_all_old_pairs_and_rejects_altered_seal(self):
        self.assertEqual(len(sources.inventory(witness())),65)
        for bad in ('count','bytes','duplicate','order','path','root','commitment'):
            value=witness()
            if bad=='count':value['files'].pop()
            elif bad=='bytes':value['files'][0]['bytes']+=1
            elif bad=='duplicate':value['files'][1]=deepcopy(value['files'][0])
            elif bad=='order':value['files'][1],value['files'][2]=value['files'][2],value['files'][1]
            elif bad=='path':value['files'][0]['path']='../outside'
            elif bad=='root':value['files'][0]['root']='elsewhere'
            else:value['commitment_sha256']='b'*64
            with self.subTest(bad=bad),self.assertRaises(ValueError):sources.inventory(value)

    def test_twelve_jobs_keep_four_original_sources_and_have_distinct_point_bindings(self):
        docs=documents();before=deepcopy(docs);plans=sources.plans(docs)
        self.assertEqual([p['role'] for p in plans],list(contract.ROLES))
        self.assertEqual(len({p['role'] for p in plans}),12)
        for doc in docs:
            chosen=[sources.selected_point(doc,q) for q in contract.POINTS]
            self.assertEqual({c['source']['role'] for c in chosen},{doc['origin']})
            self.assertEqual(len({c['role'] for c in chosen}),3)
            self.assertEqual(len({p['target_sha256'] for p in plans if p['origin']==doc['origin']}),3)
            chosen[0]['source']['role']='mutated'
        self.assertEqual(docs,before)
        self.assertFalse(any(p['point']=='Q0' for p in plans))

    def test_duplicate_or_missing_index_cannot_be_replaced_in_job_plan(self):
        docs=documents();docs[0]['quantiles']['points'][0].update(target=None,index=0,reason='duplicate_index')
        plans=sources.plans(docs)
        self.assertEqual(plans[0]['official_actions'],0);self.assertIsNone(plans[0]['target_sha256'])
        self.assertEqual(plans[0]['reason'],'duplicate_index')
        self.assertEqual(len(plans),12)
        with self.assertRaises(ValueError):sources.selected_point(docs[0],'Q0')
        docs[0]['quantiles']['points'][1]['target']['step']=239
        with self.assertRaises(ValueError):sources.plans(docs)


if __name__=='__main__':unittest.main()
