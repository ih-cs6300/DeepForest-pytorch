import argparse

# parser stuff
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument('--site', type=str, required=True, help='name of site')
parser.add_argument('--train_dir', type=str, required=True, help='training directory')
parser.add_argument('--test_dir', type=str, required=True, help='test directory')
parser.add_argument('--train_ann', type=str, required=True, help='training annotations')
parser.add_argument('--val_ann', type=str, required=True, help='validation annotations')
parser.add_argument('--test_ann', type=str, required=True, help='testing annotations')
parser.add_argument('--epochs', type=int, required=True, default=5, help='number of epochs')
parser.add_argument('--chm', type=str, required=True, default='False', help='use chm')
parser.add_argument('--log', type=str, required=True, help='local file:"log.csv" or "comet"')
parser.add_argument('--C', type=float, required=True, help='normalizing constant')
parser.add_argument('--pi_start', type=int, required=True, help='epoch to start to apply rule')
parser.add_argument('--pi_0', type=float, required=True, help='first term of pi_params')
parser.add_argument('--pi_1', type=float, required=True, help='second term of pi_params')
parser.add_argument('--pi_f', type=float, required=False, help='fixed value of pi')

args = parser.parse_args()
