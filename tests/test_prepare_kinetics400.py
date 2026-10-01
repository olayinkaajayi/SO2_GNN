import csv
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data_cleaning.prepare_kinetics400 import main


class PrepareKinetics400Tests(unittest.TestCase):
    def write_annotations(self, root):
        train_csv = root / 'train.csv'
        val_csv = root / 'val.csv'
        class_names_path = root / 'class_names.txt'
        class_names_path.write_text(
            ''.join(f'action_{index:03d}\n' for index in range(400)),
            encoding='utf-8',
        )
        fields = ['label', 'youtube_id', 'time_start', 'time_end', 'split', 'is_cc']
        train_records = [
            {
                'label': f'action_{index:03d}',
                'youtube_id': f'train_{index:03d}',
                'time_start': '0',
                'time_end': '10',
                'split': 'train',
                'is_cc': '0',
            }
            for index in range(400)
        ]
        val_record = {
            'label': 'action_399',
            'youtube_id': 'eval_sample',
            'time_start': '5',
            'time_end': '15',
            'split': 'val',
            'is_cc': '0',
        }
        for csv_path, records in ((train_csv, train_records), (val_csv, [val_record])):
            with csv_path.open('w', newline='', encoding='utf-8') as annotation_file:
                writer = csv.DictWriter(annotation_file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
        return train_csv, val_csv, class_names_path

    def test_converts_stgcn_packed_npy_and_label_pickle(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_csv, val_csv, class_names_path = self.write_annotations(root)
            train_data_path = root / 'train_data.npy'
            val_data_path = root / 'val_data.npy'
            train_labels_path = root / 'train_label.pkl'
            val_labels_path = root / 'val_label.pkl'
            output_path = root / 'kinetics_packed.npz'

            train_data = np.zeros((1, 3, 300, 18, 2), dtype=np.float32)
            val_data = np.zeros((1, 3, 300, 18, 2), dtype=np.float32)
            train_data[:, 0, [0, 2]] = -0.2
            train_data[:, 1, [0, 2]] = 0.1
            train_data[:, 2, [0, 2], :, 0] = 0.4
            train_data[:, 2, [0, 2], :, 1] = 1.05
            val_data[:] = train_data
            np.save(train_data_path, train_data)
            np.save(val_data_path, val_data)

            with train_labels_path.open('wb') as label_file:
                pickle.dump((['train_000.json'], [np.int64(0)]), label_file)
            with val_labels_path.open('wb') as label_file:
                pickle.dump((['eval_sample.json'], [np.int64(399)]), label_file)

            arguments = [
                'prepare_kinetics400.py',
                '--train-csv', str(train_csv),
                '--val-csv', str(val_csv),
                '--class-names', str(class_names_path),
                '--train-poses', str(train_data_path),
                '--val-poses', str(val_data_path),
                '--train-labels', str(train_labels_path),
                '--val-labels', str(val_labels_path),
                '--output', str(output_path),
                '--compression', 'stored',
            ]
            with patch.object(sys, 'argv', arguments):
                main()

            with np.load(output_path, allow_pickle=False) as archive:
                self.assertEqual(archive['x_train'].shape, (1, 3, 300, 18, 2))
                self.assertEqual(archive['x_eval'].shape, (1, 3, 300, 18, 2))
                self.assertEqual(archive['y_train'].tolist(), [0])
                self.assertEqual(archive['y_eval'].tolist(), [399])
                self.assertAlmostEqual(float(archive['x_train'][0, 0, 0, 0, 0]), -0.2)
                self.assertAlmostEqual(float(archive['x_train'][0, 1, 0, 0, 1]), 0.1)
                self.assertAlmostEqual(float(archive['x_train'][0, 2, 0, 0, 1]), 1.05)
                self.assertEqual(float(archive['x_train'][0, :, 1].sum()), 0.0)
                self.assertEqual(archive['class_names'].shape, (400,))

    def test_failed_conversion_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_csv, val_csv, class_names_path = self.write_annotations(root)
            train_poses = root / 'train_poses'
            val_poses = root / 'val_poses'
            train_poses.mkdir()
            val_poses.mkdir()
            output_path = root / 'existing.npz'
            original_contents = b'previous valid dataset archive'
            output_path.write_bytes(original_contents)

            arguments = [
                'prepare_kinetics400.py',
                '--train-csv', str(train_csv),
                '--val-csv', str(val_csv),
                '--class-names', str(class_names_path),
                '--train-poses', str(train_poses),
                '--val-poses', str(val_poses),
                '--output', str(output_path),
            ]
            with patch.object(sys, 'argv', arguments):
                with self.assertRaisesRegex(ValueError, 'packed ST-GCN .npy'):
                    main()

            self.assertEqual(output_path.read_bytes(), original_contents)
            self.assertEqual(list(root.glob('.existing.npz.*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
