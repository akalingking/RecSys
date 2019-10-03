#!/usr/bin/python
# -*- encoding: utf-8 -*-
import unittest
from deepfm_test import TestDeepFM
from pipeline_test import TestPipeline
from metric_test import TestMetric


def main():
    suite = unittest.TestSuite()

    # suite.addTest(TestPipeline("pipeline_test"))
    suite.addTest(TestMetric("metric_test"))
    suite.addTest(TestDeepFM("deepfm_test"))

    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    main()
