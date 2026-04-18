import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dst.analysis.eval_audit import (
    canonicalize_value,
    compute_alignment_features,
    derive_error_family,
)


class TestEvalAudit(unittest.TestCase):
    def test_canonicalize_guesthouse_and_bnb(self) -> None:
        self.assertEqual(canonicalize_value("hotel-name", "guesthouse"), "guest house")
        self.assertEqual(canonicalize_value("hotel-name", "b and b"), "bed and breakfast")
        self.assertEqual(canonicalize_value("hotel-name", "b&b"), "bed and breakfast")

    def test_canonicalize_restaurant_name_suffix(self) -> None:
        self.assertEqual(canonicalize_value("restaurant-name", "ask restaurant"), "ask")
        self.assertEqual(canonicalize_value("restaurant-name", "restaurant"), "restaurant")

    def test_alignment_features_user_turns(self) -> None:
        context = (
            "Turn 0: I want a b&b in town\n"
            "Turn 1: Sure, any preference?\n"
            "Turn 2: Book the guesthouse please"
        )
        features = compute_alignment_features(
            "hotel-name",
            "guest house",
            "guest house",
            context,
        )
        self.assertTrue(features["gold_in_full_context_canon"])
        self.assertTrue(features["gold_in_user_turns_canon"])

    def test_error_family_canonical_only(self) -> None:
        context = "Turn 0: I want a guesthouse"
        features = compute_alignment_features("hotel-name", "guest house", "guest house", context)
        label = derive_error_family(
            "hotel-name",
            "guest house",
            "guesthouse",
            "guest house",
            "guest house",
            features,
            context,
        )
        self.assertEqual(label, "correct_canonical_only")

    def test_error_family_none_to_value_carryover(self) -> None:
        context = "Turn 0: I want a bed and breakfast"
        features = compute_alignment_features("hotel-name", "none", "none", context)
        label = derive_error_family(
            "hotel-name",
            "none",
            "bed and breakfast",
            "none",
            "bed and breakfast",
            features,
            context,
        )
        self.assertEqual(label, "carryover_from_context")

    def test_error_family_dontcare_to_none(self) -> None:
        context = "Turn 0: Any is fine"
        features = compute_alignment_features("hotel-name", "dontcare", "dontcare", context)
        label = derive_error_family(
            "hotel-name",
            "dontcare",
            "none",
            "dontcare",
            "none",
            features,
            context,
        )
        self.assertEqual(label, "dontcare_to_none")

    def test_error_family_missed_explicit_value(self) -> None:
        context = "Turn 0: I need a hotel in the centre"
        features = compute_alignment_features("hotel-area", "centre", "centre", context)
        label = derive_error_family(
            "hotel-area",
            "centre",
            "none",
            "centre",
            "none",
            features,
            context,
        )
        self.assertEqual(label, "missed_explicit_value")


if __name__ == "__main__":
    unittest.main()
