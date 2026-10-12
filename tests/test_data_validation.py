"""Small CPU checks; never calls the training loop."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from train_compare import make_dataset, standardize_train_test, BaselineNet, ImprovedNet, count_parameters

class DataChecks(unittest.TestCase):
    def test_default_split_shape(self):
        train, labels, test, test_labels = make_dataset()
        self.assertEqual(tuple(train.shape), (900, 2))
        self.assertEqual(tuple(test.shape), (300, 2))
        self.assertEqual(len(labels), 900)
        self.assertEqual(len(test_labels), 300)

    def test_empty_test_split_is_rejected(self):
        with self.assertRaises(ValueError): make_dataset(20, 0.001)

    def test_too_small_train_split_is_rejected(self):
        with self.assertRaises(ValueError): make_dataset(20, 0.999)

    def test_standardization_uses_training_statistics(self):
        train = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        extreme_test = torch.tensor([[1000., -1000.]])
        standardized, test, mean, std = standardize_train_test(train, extreme_test)
        torch.testing.assert_close(mean, train.mean(0, keepdim=True))
        torch.testing.assert_close(standardized.mean(0), torch.zeros(2))
        torch.testing.assert_close(test, (extreme_test - mean) / std)

    def test_documented_parameter_counts(self):
        self.assertEqual(count_parameters(BaselineNet()), 42)
        self.assertEqual(count_parameters(ImprovedNet()), 1218)

if __name__ == '__main__': unittest.main()
