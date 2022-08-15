import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--room1', help='File path for first room impulse response.')
parser.add_argument('--room2', help='File path for second room impulse response. Should be a legitimate device that should authenticate with room1.')
parser.add_argument('--room3', help='File path for third room impulse response. Should be an adversarial case, in other words, a different location than room1.')
parser.add_argument('--original_audio', help='Path to a untainted wav file that the room files listen to within the room.')
parser.add_argument('--align_buffers', action='store_true', help="Argument that specifies if the data is aligned or not. Default value is false and the code will align the buffer." )

args = vars(parser.parse_args())
