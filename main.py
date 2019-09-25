from kinnet import get_dataset, check_downloads_directory
import argparse


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Download Datasets')
    parser.add_argument('-n', metavar='--datasetnum', type=int,
                        help='dataset Number')
    parser.add_argument("-d", metavar = "--downloadDir", type=str,
                            help="Directory to save downloaded files")
    args = parser.parse_args()
    get_dataset(args.n, args.d)