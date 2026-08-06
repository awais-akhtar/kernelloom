from __future__ import annotations

import unittest

from kernelloom.cpu import available_cpu_cores, plan_cpu_execution


class CPUPlanTests(unittest.TestCase):
    def test_profiles_respect_available_cores_and_have_distinct_intent(self) -> None:
        latency = plan_cpu_execution("latency", available_cores=8, reserve_cores=2)
        throughput = plan_cpu_execution("throughput", available_cores=8, reserve_cores=2)
        efficient = plan_cpu_execution("efficient", available_cores=8, reserve_cores=2)
        self.assertEqual(latency.threads, 6)
        self.assertEqual(throughput.threads, 8)
        self.assertEqual(throughput.reserved_cores, 0)
        self.assertLess(efficient.threads, latency.threads)
        self.assertGreater(throughput.recommended_batch_size, latency.recommended_batch_size)

    def test_auto_and_invalid_plans(self) -> None:
        self.assertEqual(plan_cpu_execution("auto", available_cores=1).threads, 1)
        self.assertGreaterEqual(available_cpu_cores(), 1)
        with self.assertRaises(ValueError):
            plan_cpu_execution("turbo")
        with self.assertRaises(ValueError):
            plan_cpu_execution("auto", reserve_cores=-1)


if __name__ == "__main__":
    unittest.main()
