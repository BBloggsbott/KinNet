from kinnet import get_dataset, check_downloads_directory, KinNetDataset, KinNetTrainer
import argparse


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Download Datasets')
    parser.add_argument('-n', metavar='--datasetnum', type=int,
                        help='dataset Number')
    parser.add_argument("-d", metavar = "--downloadDir", type=str,
                            help="Directory to save downloaded files")
    parser.add_argument("-bs", metavar = "--batchSize", type=int,
                            help="Batch Size for training")
    parser.add_argument("-e", metavar = "--epochs", type=int,
                            help="Number of epochs for training")
    args = parser.parse_args()
    trainer = KinNetTrainer(args.n, args.d, args.bs)
    trainer.data.show_batch()
    trainer.train_model(args.e)
