"""Download the official Kinetics-400 train and validation annotation CSVs."""

import argparse
from pathlib import Path
from urllib.request import urlopen


ANNOTATION_URLS = {
    'train.csv': 'https://s3.amazonaws.com/kinetics/400/annotations/train.csv',
    'val.csv': 'https://s3.amazonaws.com/kinetics/400/annotations/val.csv',
    'stgcn_label_names.txt': (
        'https://raw.githubusercontent.com/yysijie/st-gcn/master/'
        'resource/kinetics_skeleton/label_name.txt'
    ),
}


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + '.part')
    try:
        with urlopen(url, timeout=60) as response, temporary_path.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/kinetics400/annotations'),
        help='Directory where official train.csv and val.csv are stored.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Redownload files even when they already exist.',
    )
    args = parser.parse_args()

    for filename, url in ANNOTATION_URLS.items():
        destination = args.output_dir / filename
        if destination.exists() and not args.force:
            print(f'Already present: {destination}')
            continue
        download_file(url, destination)
        print(f'Downloaded {url} -> {destination}')


if __name__ == '__main__':
    main()
