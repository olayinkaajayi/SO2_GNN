"""Package ST-GCN OpenPose-18 Kinetics-400 splits in the training NPZ format."""

import argparse
import csv
import importlib
import logging
import pickle
from pathlib import Path
import tempfile
import zipfile

import numpy as np


EXPECTED_FIELDS = {'label', 'youtube_id', 'time_start', 'time_end', 'split'}
EXPECTED_CLASSES = 400
STGCN_FRAMES = 300
STGCN_JOINTS = 18
STGCN_PEOPLE = 2


def sample_id(record):
    start = int(float(record['time_start']))
    end = int(float(record['time_end']))
    return f"{record['youtube_id']}_{start}_{end}"


def read_annotations(path, expected_split):
    records = []
    with path.open(newline='', encoding='utf-8') as annotation_file:
        reader = csv.DictReader(annotation_file)
        if reader.fieldnames is None or not EXPECTED_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f'{path} must contain columns: {sorted(EXPECTED_FIELDS)}')
        for row in reader:
            if row['split'] != expected_split:
                raise ValueError(
                    f'{path} contains split={row["split"]!r}; expected {expected_split!r}'
                )
            if not row['label'] or not row['youtube_id']:
                raise ValueError(f'{path} contains an empty label or youtube_id')
            records.append(row)
    if not records:
        raise ValueError(f'{path} contains no annotation records')
    ids = [sample_id(record) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{path} contains duplicate video/time sample IDs')
    return records


def read_class_names(path):
    with path.open(encoding='utf-8') as class_file:
        classes = [line.strip() for line in class_file if line.strip()]
    if (len(classes) != EXPECTED_CLASSES or len(set(classes)) != EXPECTED_CLASSES):
        raise ValueError(f'{path} must contain exactly 400 unique class names in source order')
    return classes


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == 'scalar':
            return importlib.import_module(module).scalar
        if module == 'numpy' and name == 'dtype':
            return np.dtype
        raise pickle.UnpicklingError(f'Unsupported global in trusted ST-GCN label file: {module}.{name}')


def read_stgcn_label_file(path):
    with path.open('rb') as label_file:
        sample_names, source_labels = RestrictedUnpickler(label_file, encoding='latin1').load()
    if len(sample_names) != len(source_labels):
        raise ValueError(f'{path}: sample names and labels have different lengths')
    return sample_names, source_labels


def stgcn_array_to_pose(sample):
    if sample.shape != (3, STGCN_FRAMES, STGCN_JOINTS, STGCN_PEOPLE):
        raise ValueError(
            'ST-GCN samples must have shape [3, 300, 18, 2], '
            f'got {sample.shape}'
        )
    pose = np.asarray(sample, dtype=np.float32)
    if not np.isfinite(pose).all():
        raise ValueError('ST-GCN pose sample contains NaN or infinite values')
    if np.any(pose[2] < 0):
        raise ValueError('ST-GCN confidence values must be non-negative')
    return pose


def write_npy_member(archive, name, shape, dtype, chunks, compression):
    dtype = np.dtype(dtype)
    info = zipfile.ZipInfo(f'{name}.npy')
    info.compress_type = compression
    header = {
        'descr': np.lib.format.dtype_to_descr(dtype),
        'fortran_order': False,
        'shape': shape,
    }
    with archive.open(info, mode='w', force_zip64=True) as output:
        np.lib.format.write_array_header_1_0(output, header)
        for chunk in chunks:
            chunk = np.ascontiguousarray(chunk, dtype=dtype)
            output.write(memoryview(chunk).cast('B'))


def convert_split_from_stgcn_npy(archive, split, data_path, label_path, records,
                                 class_to_id, compression):
    source_data = np.load(data_path, mmap_mode='r', allow_pickle=False)
    if (source_data.ndim != 5 or source_data.shape[1] != 3 or
            source_data.shape[2:] != (STGCN_FRAMES, STGCN_JOINTS, STGCN_PEOPLE)):
        raise ValueError(
            f'{data_path} must have shape [N, 3, 300, 18, 2], got {source_data.shape}'
        )
    sample_names, source_labels = read_stgcn_label_file(label_path)
    if len(sample_names) != source_data.shape[0]:
        raise ValueError(f'{data_path} and {label_path} contain different sample counts')

    annotations_by_id = {}
    for record in records:
        video_id = record['youtube_id']
        if video_id in annotations_by_id:
            raise ValueError(
                f'{data_path}: duplicate official annotation rows for video ID {video_id}'
            )
        annotations_by_id[video_id] = record
    selected = []
    missing_annotations = 0
    for source_index, sample_name in enumerate(sample_names):
        if isinstance(sample_name, bytes):
            sample_name = sample_name.decode('utf-8')
        identifier = Path(str(sample_name)).stem
        record = annotations_by_id.get(identifier)
        if record is None:
            missing_annotations += 1
            continue
        expected_label = class_to_id[record['label']]
        if int(source_labels[source_index]) != expected_label:
            raise ValueError(
                f'{label_path}: source label for {identifier} does not match official annotations'
            )
        selected.append((source_index, record))

    if not selected:
        raise ValueError(f'No ST-GCN samples from {data_path} match official annotations')
    if missing_annotations:
        logging.warning(
            '%s: skipping %d source samples without official annotations',
            split, missing_annotations,
        )

    def pose_chunks():
        batch = []
        for index, (source_index, _) in enumerate(selected, start=1):
            batch.append(stgcn_array_to_pose(
                source_data[source_index]
            ))
            if len(batch) == 64:
                yield np.stack(batch)
                batch.clear()
            if index % 1000 == 0 or index == len(selected):
                logging.info('%s: converted %d/%d packed pose tracks', split, index, len(selected))
        if batch:
            yield np.stack(batch)

    write_npy_member(
        archive, f'x_{split}', (len(selected), 3, STGCN_FRAMES, STGCN_JOINTS, STGCN_PEOPLE),
        np.float32, pose_chunks(), compression,
    )
    write_npy_member(
        archive,
        f'y_{split}',
        (len(selected),),
        np.int64,
        [np.asarray([class_to_id[record['label']] for _, record in selected], dtype=np.int64)],
        compression,
    )
    return len(selected), missing_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', type=Path, default=Path('data/kinetics400/annotations/train.csv'))
    parser.add_argument('--val-csv', type=Path, default=Path('data/kinetics400/annotations/val.csv'))
    parser.add_argument(
        '--class-names', type=Path,
        default=Path('data/kinetics400/annotations/stgcn_label_names.txt'),
        help='Canonical ordered Kinetics class names from the ST-GCN source release.',
    )
    parser.add_argument('--train-poses', type=Path, required=True)
    parser.add_argument('--val-poses', type=Path, required=True)
    parser.add_argument('--train-labels', type=Path)
    parser.add_argument('--val-labels', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/kinetics400/kinetics400_openpose18.npz'))
    parser.add_argument(
        '--compression', choices=('stored', 'deflated'), default='deflated',
        help='Use stored for faster conversion and larger output; deflated saves disk space.',
    )
    args = parser.parse_args()

    train_records = read_annotations(args.train_csv, 'train')
    eval_records = read_annotations(args.val_csv, 'val')
    classes = read_class_names(args.class_names)
    annotation_classes = {record['label'] for record in train_records + eval_records}
    if annotation_classes != set(classes):
        raise ValueError('Official CSV labels do not match the ST-GCN class-name list')
    class_to_id = {class_name: index for index, class_name in enumerate(classes)}
    compression = zipfile.ZIP_STORED if args.compression == 'stored' else zipfile.ZIP_DEFLATED

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            prefix=f'.{args.output.name}.', suffix='.tmp', dir=args.output.parent, delete=False
    ) as temporary_file:
        temporary_output = Path(temporary_file.name)

    try:
        with zipfile.ZipFile(temporary_output, mode='w', compression=compression, allowZip64=True) as archive:
            split_sources = (
                ('train', args.train_poses, args.train_labels, train_records),
                ('eval', args.val_poses, args.val_labels, eval_records),
            )
            counts = []
            for split, pose_path, label_path, records in split_sources:
                if not pose_path.is_file() or pose_path.suffix.lower() != '.npy':
                    raise ValueError(f'{pose_path} must be a packed ST-GCN .npy pose array')
                if label_path is None:
                    label_path = pose_path.with_name(
                        pose_path.name.replace('_data.npy', '_label.pkl')
                    )
                counts.append(convert_split_from_stgcn_npy(
                    archive, split, pose_path, label_path, records, class_to_id,
                    compression,
                ))
            (train_count, train_skipped), (eval_count, eval_skipped) = counts
            write_npy_member(
                archive,
                'class_names',
                (len(classes),),
                np.dtype(f'<U{max(map(len, classes))}'),
                [np.asarray(classes)],
                compression,
            )
        temporary_output.replace(args.output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    logging.info(
        'Wrote %s (%d train, %d eval; skipped %d train and %d eval source samples)',
        args.output, train_count, eval_count, train_skipped, eval_skipped,
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
