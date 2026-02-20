import json
import os
import subprocess
import tempfile
import unittest


class TestCheckImputationThresholds(unittest.TestCase):
    def test_checker_accepts_good_report(self):
        rpt = {
            "rows_with_any_imputation_prop": 0.0,
            "per_column_missing_rate": {"x": 0.0}
        }
        fd, path = tempfile.mkstemp(text=True)
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rpt, fh)

        res = subprocess.run(["python", "scripts/check_imputation_thresholds.py", "--report", path, "--max-row-prop", "0.1", "--max-col-rate", "0.1"], check=False)
        os.remove(path)
        self.assertEqual(res.returncode, 0)

    def test_checker_rejects_bad_report(self):
        rpt = {
            "rows_with_any_imputation_prop": 1.0,
            "per_column_missing_rate": {"x": 0.9}
        }
        fd, path = tempfile.mkstemp(text=True)
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rpt, fh)

        res = subprocess.run(["python", "scripts/check_imputation_thresholds.py", "--report", path, "--max-row-prop", "0.1", "--max-col-rate", "0.1"], check=False)
        os.remove(path)
        self.assertNotEqual(res.returncode, 0)


if __name__ == '__main__':
    unittest.main()
