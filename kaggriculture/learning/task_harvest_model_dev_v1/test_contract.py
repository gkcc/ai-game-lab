"""Keep actual per-unit execution evidence out of the public result contract."""
import unittest

from .contract import public_choice


class Projection(unittest.TestCase):
    def fixture(self):
        return dict(counts=dict(callbacks=4,callbacks_with_added_candidates=2,
            added_candidates=2,selected_virtual_harvest=1,private_identifier=98),
            selected_virtual_actual_executions=1,selected_virtual_harvested_melon_units=5,
            selected_virtual_execution_status_counts={'executed':1},
            selected_virtual_effects=[dict(step=241,unit=0,target=[4,4],
                inventory={'MELON':5},tile={'crop':'MELON'})],
            secret='synthetic-not-publishable')

    def test_explicit_aggregate_only_projection(self):
        actual=public_choice(self.fixture())
        self.assertEqual(actual,dict(counts=dict(callbacks=4,callbacks_with_added_candidates=2,
            added_candidates=2,selected_virtual_harvest=1),selected_virtual_actual_executions=1,
            selected_virtual_harvested_melon_units=5,
            selected_virtual_execution_status_counts={'executed':1}))

    def test_type_confused_or_unknown_status_rejects(self):
        for key,value in [('selected_virtual_actual_executions',True),
                          ('selected_virtual_execution_status_counts',{'private_payload':1})]:
            item=self.fixture();item[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):public_choice(item)


if __name__=='__main__':unittest.main()
