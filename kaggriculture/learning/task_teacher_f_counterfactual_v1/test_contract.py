"""Synthetic invariants for missing coverage and credible screening counts."""
from copy import deepcopy
import unittest
from . import contract as c


def screen_fixture():
    result=[]
    for source in c.SOURCES:
        coverage=dict(candidate_found=False,status_by_window=[['empty','empty'] for _ in c.WINDOWS],
            source_trace_lines=719,comparisons=0,official_transitions_reserved=0,
            meter=dict(comparisons_reserved=0,comparisons_begun=0,comparisons_completed=0,
                controller_turns_reserved=0,controller_turns_actual='unknown',
                official_transitions_reserved=0,official_helpers_begun=0,official_helpers_completed=0,
                official_transitions_confirmed_completed=0))
        result.append(dict(source=dict(source),coverage=coverage,candidate=None))
    return result


def hit(item):
    item['candidate']=dict(role=item['source']['role'],opponent=item['source']['opponent'],
        side=item['source']['side'],window=0,rank=0)
    row=item['coverage'];row['candidate_found']=True;row['status_by_window'][0][0]='official_next_frame_different'
    row['comparisons']=1;row['official_transitions_reserved']=2
    row['meter'].update(comparisons_reserved=1,comparisons_begun=1,comparisons_completed=1,
        controller_turns_reserved=2,official_transitions_reserved=2,official_helpers_begun=1,
        official_helpers_completed=1,official_transitions_confirmed_completed=2)


def account(delta):
    before=dict(own=100.,opponent=50.,margin=50.,score=1.)
    after=dict(own=100.+delta,opponent=50.,margin=50.+delta,score=1.)
    return dict(control=before,intervention=after,delta={k:after[k]-before[k] for k in c.FIELDS})


class ContractTests(unittest.TestCase):
    def test_fixed_empty_screen_is_a_valid_negative_with_no_invented_return(self):
        result=c.screen_projection(screen_fixture())
        self.assertEqual(result['candidate_count'],0)
        self.assertEqual(result['represented_seed_clusters'],0)
        self.assertEqual(result['source_trace_lines'],5752)
        self.assertEqual(result['conclusion'],'no_measurable_candidate_in_fixed_screen')
        self.assertFalse(result['future_returns_evaluated'])

    def test_successful_hit_requires_a_real_counted_comparison_and_two_steps(self):
        values=screen_fixture();hit(values[0])
        result=c.screen_projection(values)
        self.assertEqual(result['candidate_count'],1)
        self.assertEqual(result['counters']['official_transitions_confirmed_completed'],2)
        broken=deepcopy(values)
        broken[0]['coverage']['meter']=screen_fixture()[0]['coverage']['meter']
        broken[0]['coverage'].update(comparisons=0,official_transitions_reserved=0)
        with self.assertRaises(ValueError):c.screen_projection(broken)

    def test_unsupported_status_or_execution_after_first_hit_is_not_a_valid_certificate(self):
        values=screen_fixture();values[0]['coverage']['status_by_window'][0][0]='unsupported_status'
        with self.assertRaises(ValueError):c.screen_projection(values)
        values=screen_fixture();hit(values[0]);values[0]['coverage']['status_by_window'][0][1]='request_same'
        row=values[0]['coverage'];row['comparisons']=2
        row['meter'].update(comparisons_reserved=2,comparisons_begun=2,comparisons_completed=2,controller_turns_reserved=4)
        with self.assertRaises(ValueError):c.screen_projection(values)

    def test_missing_source_trace_or_wrong_candidate_slot_is_rejected(self):
        values=screen_fixture();values[2]['coverage']['source_trace_lines']=718
        with self.assertRaises(ValueError):c.screen_projection(values)
        values=screen_fixture();hit(values[0]);values[0]['candidate']['rank']=1
        with self.assertRaises(ValueError):c.screen_projection(values)
        values=screen_fixture();values[1]=deepcopy(values[0])
        with self.assertRaises(ValueError):c.screen_projection(values)

    def test_missing_seed_groups_are_not_zero_filled_and_weighting_is_explicit(self):
        accounts=[account(12),account(12),account(-6)]
        result=c.pair_projection(accounts,list(c.SOURCES[:3]))
        self.assertEqual(result['case_mean_delta']['own'],6.)
        self.assertEqual(result['seed_mean_delta']['own'],3.)
        self.assertEqual(result['represented_seed_clusters'],2)
        self.assertEqual(result['positive_zero_negative']['own'],dict(positive=2,zero=0,negative=1))
        self.assertFalse(result['promotion'])
        empty=c.pair_projection([],[])
        self.assertIsNone(empty['case_mean_delta'])
        self.assertIsNone(empty['seed_mean_delta'])

    def test_pair_cash_and_duplicate_coordinates_cannot_be_forged(self):
        bad=account(12);bad['delta']['own']=20
        with self.assertRaises(ValueError):c.pair_projection([bad],[c.SOURCES[0]])
        with self.assertRaises(ValueError):c.pair_projection([account(1),account(1)],[c.SOURCES[0],c.SOURCES[0]])
        bad_source=dict(c.SOURCES[0],side=False)
        with self.assertRaises(ValueError):c.source_index(bad_source)


if __name__=='__main__':unittest.main()
