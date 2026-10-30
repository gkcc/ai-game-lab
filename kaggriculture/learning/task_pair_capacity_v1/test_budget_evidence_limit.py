"""Show why a traceback alone cannot authenticate a budget failure.

These are synthetic, in-memory counterexamples.  They do not identify the cause
of any past or future game failure and are not a production classifier.
"""
import dis
import unittest

from learning.task_request_gate_v1.policy import _Budget


def caught(call):
    try:
        call()
    except Exception as error:
        return error
    raise AssertionError('Synthetic call did not fail')


def exhausted():
    budget = _Budget(dict(task_forwards=1))
    budget.forward('task', 1)
    budget.forward('task', 1)


def last_frame(error):
    tb = error.__traceback__
    if tb is None:
        raise AssertionError('Synthetic traceback missing')
    while tb.tb_next is not None:
        tb = tb.tb_next
    return tb


class BudgetEvidenceLimitTests(unittest.TestCase):
    def test_forged_exception_can_share_the_exact_budget_traceback(self):
        original = caught(exhausted)
        forged = RuntimeError('synthetic unrelated error').with_traceback(original.__traceback__)
        self.assertIsNot(original, forged)
        self.assertIs(type(original), RuntimeError)
        self.assertIs(type(forged), RuntimeError)
        self.assertIs(forged.__traceback__, original.__traceback__)
        actual, attached = last_frame(original), last_frame(forged)
        self.assertIs(actual, attached)
        self.assertIs(actual.tb_frame.f_code, _Budget.forward.__code__)
        instruction = next(i for i in dis.get_instructions(_Budget.forward)
                           if i.offset == actual.tb_lasti)
        self.assertEqual((instruction.opname, instruction.arg), ('RAISE_VARARGS', 1))
        self.assertIs(type(actual.tb_frame.f_locals['self']), _Budget)
        self.assertEqual((actual.tb_frame.f_locals['kind'], actual.tb_frame.f_locals['rows']),
                         ('task', 1))
        # Identical type, frame, bytecode offset and locals do not prove which
        # exception object was actually raised at that frame.

    def test_wrapping_and_finally_can_hide_the_original_direct_traceback(self):
        def wrapped():
            try:
                exhausted()
            except RuntimeError:
                raise RuntimeError('synthetic wrapper') from None

        def failed_receipt():
            try:
                exhausted()
            finally:
                raise OSError('synthetic receipt failure')

        outer = caught(wrapped)
        self.assertIs(type(outer), RuntimeError)
        self.assertTrue(outer.__suppress_context__)
        self.assertIsNot(last_frame(outer).tb_frame.f_code, _Budget.forward.__code__)
        self.assertIs(type(outer.__context__), RuntimeError)
        self.assertIs(last_frame(outer.__context__).tb_frame.f_code, _Budget.forward.__code__)

        final = caught(failed_receipt)
        self.assertIs(type(final), OSError)
        self.assertIs(type(final.__context__), RuntimeError)
        self.assertIs(last_frame(final.__context__).tb_frame.f_code, _Budget.forward.__code__)


if __name__ == '__main__':
    unittest.main()
