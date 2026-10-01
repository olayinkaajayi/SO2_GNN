# Kinetics-400 OpenPose-18 Input

The training code consumes pose sequences, not Kinetics video files. The official Kinetics distribution provides videos and annotations; the ST-GCN authors separately released OpenPose skeleton tracks for Kinetics. This project uses those tracks when available, so pose estimation over the full video set is not required.

The official Kinetics-400 train and validation annotation CSVs and ST-GCN's ordered class-name list are downloaded locally to `data/kinetics400/annotations/`. The existing `data/` ignore rule keeps dataset files out of version control. Refresh or fetch all three metadata files with:

```bash
python data_cleaning/download_kinetics400_annotations.py
```

The ST-GCN authors describe their released Kinetics-skeleton data in the [ST-GCN README](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md#kinetics-skeleton). Their processed-data archive is downloadable from Google Drive with:

```bash
python -m pip install gdown
python data_cleaning/download_stgcn_kinetics_skeletons.py
```

The archive is about 8.64 GB and the transfer can be resumed by running the command again. The separate [CVDF Kinetics-400 repository](https://github.com/cvdfoundation/kinetics-dataset) provides video archives and annotations; downloading the full video set is optional and requires rerunning pose estimation.

After downloading the archive, extract it under the ignored `data/` directory. The archive contains packed ST-GCN `train_data.npy` / `val_data.npy` arrays with corresponding `train_label.pkl` / `val_label.pkl` files. These arrays already use OpenPose-18, two people, 300 frames, centered x/y, and confidence scores. Package them without changing their joint/person axes or score values:

```bash
unzip -q data/kinetics400/stgcn_processed_data.zip \
  'data/Kinetics/kinetics-skeleton/*' -d data/kinetics400/stgcn
python data_cleaning/prepare_kinetics400.py \
  --train-poses data/kinetics400/stgcn/data/Kinetics/kinetics-skeleton/train_data.npy \
  --val-poses data/kinetics400/stgcn/data/Kinetics/kinetics-skeleton/val_data.npy \
  --compression stored
```

Label-pickle paths default to matching sibling names (`train_label.pkl` and `val_label.pkl`); override them with `--train-labels` and `--val-labels` if needed. The converter safely reads the ST-GCN sample-name/label tuples, memory-maps the large pose arrays, matches video IDs to official CSV rows, checks source label indices against ST-GCN's ordered class names and official labels, and skips only unmatched or entirely empty source tracks. Output labels retain canonical ST-GCN class indices. Conversion streams to an atomic temporary archive; `--compression stored` produces a larger file but may load faster.

## Archive Contract

Set `dataset: kinetics400` and point `dataset_args.data_path` to one NumPy NPZ archive containing:

| Key | Required shape | Meaning |
| --- | --- | --- |
| `x_train` | `[N, 3, 300, 18, 2]` | Training poses, float channels |
| `y_train` | `[N]` | Training labels, integer class IDs 0 through 399 |
| `x_eval` | `[N, 3, 300, 18, 2]` | Evaluation poses, same representation |
| `y_eval` | `[N]` | Evaluation labels, integer class IDs 0 through 399 |
| `class_names` | `[400]` | Unique class names in the exact integer-label order used above |

The axes are sample, channel, time, OpenPose-18 joint, and person. Channel 0 is centered x, channel 1 is centered y, and channel 2 is the original OpenPose confidence score. The graph is `openpose18`, with the ST-GCN 18-joint edge list; there are at most two people per frame. The converter preserves the packed ST-GCN tensor, including person ordering, confidence values, and empty frames. In this project confidence is intentionally retained as channel 3 per the requested experiment; the SO(2) model will therefore receive it as its third input component.

The skeleton topology and input representation match ST-GCN, but the graph operator is not identical: this Trans-GCN path uses one normalized 18x18 adjacency matrix, while ST-GCN's spatial strategy partitions adjacency into root/close/further subsets. Treat results as matched-input comparisons, not an exact reproduction of ST-GCN's graph-convolution operator.

For a direct ST-GCN protocol comparison, keep `window_size: 300` and do not enable velocity derivation or other input transforms. The loader returns the stored 300 frames unchanged, including internal zero frames.

The converter uses the ordered ST-GCN `label_name.txt` as the authoritative index-to-class mapping and verifies it contains the same 400 labels as the official train/validation CSVs. The CSVs provide video ID, time span, split, and label metadata.

The prepared archive retains all 240,436 training and 19,796 validation samples from the packed ST-GCN splits; every sample matched an official annotation, including any all-zero pose tracks.

## Running

After creating `./data/kinetics400/kinetics400_openpose18.npz` to match the contract:

```bash
python main.py --config config/kinetics400/kinetics400.yaml
```

The official source CSVs are downloaded locally under `data/kinetics400/annotations/`; the skeleton archive and video files are not tracked by git. Verify current availability and terms when obtaining the ST-GCN archive or original videos.