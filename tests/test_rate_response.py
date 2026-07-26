import hashlib
import json
import unittest

from atic.rate_response import DEFAULT_VARIANT, build_report


def _rows(high_bpp=0.08, high_estimated=0.075, high_psnr=14.5):
    provenance = {
        "git": {"commit": "abc123", "is_dirty": False},
        "data": {"bundle_id": "bundle"},
        "run": {"initial_state_sha256": "initial"},
    }
    provenance_sha256 = hashlib.sha256(
        json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return [
        {
            "variant": DEFAULT_VARIANT,
            "seed": 42,
            "lambda_rd": 0.0018,
            "BPP_actual": 0.04,
            "BPP_estimated": 0.038,
            "PSNR": 13.0,
            "provenance": provenance,
            "provenance_sha256": provenance_sha256,
        },
        {
            "variant": DEFAULT_VARIANT,
            "seed": 42,
            "lambda_rd": 0.013,
            "BPP_actual": high_bpp,
            "BPP_estimated": high_estimated,
            "PSNR": high_psnr,
            "provenance": provenance,
            "provenance_sha256": provenance_sha256,
        },
    ]


class RateResponseTests(unittest.TestCase):
    def _report(self, rows):
        return build_report(
            rows,
            variant=DEFAULT_VARIANT,
            seed=42,
            low_lambda=0.0018,
            high_lambda=0.013,
            min_delta_bpp=0.01,
            min_delta_psnr=0.25,
        )

    def test_predeclared_rate_response_passes(self):
        report = self._report(_rows())

        self.assertTrue(report["passed"])
        self.assertTrue(report["test_locked"])
        self.assertAlmostEqual(report["delta_BPP_actual"], 0.04)
        self.assertAlmostEqual(report["delta_PSNR"], 1.5)

    def test_actual_bpp_margin_is_enforced(self):
        report = self._report(_rows(high_bpp=0.045))

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["actual_bpp_margin"])

    def test_estimated_bpp_order_is_enforced(self):
        report = self._report(_rows(high_estimated=0.03))

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["estimated_bpp_order"])

    def test_psnr_margin_is_enforced(self):
        report = self._report(_rows(high_psnr=13.1))

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["psnr_margin"])

    def test_mismatched_study_provenance_is_rejected(self):
        rows = _rows()
        rows[1]["provenance"] = {
            **rows[1]["provenance"],
            "git": {"commit": "different", "is_dirty": False},
        }
        rows[1]["provenance_sha256"] = hashlib.sha256(
            json.dumps(
                rows[1]["provenance"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        with self.assertRaisesRegex(RuntimeError, "mismatched commits"):
            self._report(rows)

    def test_duplicate_seed_lambda_is_rejected(self):
        rows = _rows()
        rows.append(dict(rows[0]))

        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            self._report(rows)

    def test_requires_exact_predeclared_pair(self):
        with self.assertRaisesRegex(RuntimeError, "Expected exactly"):
            self._report(_rows()[:1])


if __name__ == "__main__":
    unittest.main()
