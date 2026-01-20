import unittest

from src import simulator, preprocessing, feature_extraction, model


class PipelineSmokeTest(unittest.TestCase):
    def test_ecg_pipeline_smoke(self):
        ecg = simulator.generate_ecg(2.0, fs=250)
        self.assertEqual(len(ecg), 500)
        bp = preprocessing.simple_bandpass(ecg, low=0.5, high=40.0, fs=250)
        feats = feature_extraction.summary_ecg_features(bp, fs=250)
        self.assertIn("avg_hr_bpm", feats)
        ann = model.detect_anomalies(feats, eeg_bandpower=0.5)
        self.assertIn("cardiac_anomaly_score", ann)

    def test_eeg_simulator(self):
        eeg = simulator.generate_eeg(2.0, fs=128)
        self.assertEqual(len(eeg), 256)
        bp = feature_extraction.simple_bandpower(eeg, band=(8, 12), fs=128)
        self.assertIsInstance(bp, float)


if __name__ == "__main__":
    unittest.main()
