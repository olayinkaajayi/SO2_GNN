import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset import create


class Kinetics400FeederTests(unittest.TestCase):
    def create_archive(self, path, labels=None):
        data = np.zeros((2, 3, 8, 18, 2), dtype=np.float32)
        data[:, 0, :5, 0, :] = -0.25
        data[:, 1, :5, 0, :] = 0.1
        data[:, 2, :5, 0, :] = 0.8
        if labels is None:
            labels = np.array([0, 399], dtype=np.int64)
        np.savez(
            path,
            x_train=data,
            y_train=labels,
            x_eval=data[:1],
            y_eval=np.array([399], dtype=np.int64),
            class_names=np.array([f'class_{index}' for index in range(400)]),
        )

    def test_loads_train_and_eval_with_model_input_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / 'kinetics.npz'
            self.create_archive(archive_path)

            feeders, data_shape, num_classes, adjacency, _ = create(
                'kinetics400',
                graph='openpose18',
                labeling='spatial',
                data_path=str(archive_path),
                window_size=8,
                p_interval=[1.0],
                p_interval_test=[1.0],
            )

            self.assertEqual(num_classes, 400)
            self.assertEqual(data_shape, [1, 3, 8, 18, 2])
            self.assertEqual(adjacency.shape, (18, 18))
            train_pose, train_label = feeders['train'][0]
            eval_pose, eval_label = feeders['eval'][0]
            self.assertEqual(tuple(train_pose.shape), (1, 3, 8, 18, 2))
            self.assertEqual(tuple(eval_pose.shape), (1, 3, 8, 18, 2))
            self.assertEqual(train_label, 0)
            self.assertEqual(eval_label, 399)

    def test_rejects_out_of_range_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / 'kinetics.npz'
            self.create_archive(archive_path, labels=np.array([0, 400], dtype=np.int64))

            with self.assertRaisesRegex(ValueError, r'outside \[0, 399\]'):
                create(
                    'kinetics400',
                    graph='openpose18',
                    labeling='spatial',
                    data_path=str(archive_path),
                    window_size=8,
                )


if __name__ == '__main__':
    unittest.main()