"""The CI axiom audit must reject incomplete or unexpected proof dependencies."""

import unittest

from verify import audit_axioms


class AxiomAuditTests(unittest.TestCase):
    def test_accept_standard_axioms_and_axiom_free_proof(self):
        audit_axioms("'QuantizationProof.core' does not depend on any axioms\n"
                     "'QuantizationProof.concrete' depends on axioms: [propext,\n"
                     " Classical.choice, Quot.sound]\n", ['core', 'concrete'])

    def test_reject_sorry_and_custom_axioms(self):
        for dependency in ('sorryAx', 'unproved_hardware_assumption'):
            with self.subTest(dependency=dependency), self.assertRaises(AssertionError):
                audit_axioms(f"'QuantizationProof.core' depends on axioms: [propext, {dependency}]",
                             ['core'])

    def test_missing_report_does_not_silently_pass(self):
        with self.assertRaises(AssertionError):
            audit_axioms('', ['core'])


if __name__ == '__main__':
    unittest.main()
