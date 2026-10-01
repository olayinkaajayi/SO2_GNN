"""Download the ST-GCN authors' processed skeleton archive from Google Drive."""

import argparse
from pathlib import Path


STGCN_PROCESSED_DATA_ID = '103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/kinetics400/stgcn_processed_data.zip'),
        help='Destination for the approximately 8.64 GB processed-data archive.',
    )
    args = parser.parse_args()

    try:
        import gdown
    except ImportError as error:
        raise SystemExit(
            'Install gdown into the active Python environment first: python -m pip install gdown'
        ) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(
        f'https://drive.google.com/uc?id={STGCN_PROCESSED_DATA_ID}',
        str(args.output),
        quiet=False,
        resume=True,
    )
    if result is None:
        raise SystemExit('Google Drive download failed; retry with the same command to resume.')
    print(f'Downloaded processed skeleton archive to {result}')


if __name__ == '__main__':
    main()
