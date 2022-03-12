import argparse
import images_io
import transform

# Static configurations
crop_size = [0, 0, 80, 0]

parser = argparse.ArgumentParser()
parser.add_argument('--normalize', action='store_true', default=False)
parser.add_argument('--size', type=int, default=64, help='Output image size')
parser.add_argument('--stride', type=int, default=1,
                    help='Stride along the input image')
parser.add_argument('--crop-only', action='store_true', default=False,
                    help='If set, only crop images')
parser.add_argument('--threshold', type=float, default=1.0,
                    help='Maximum ratio of the highest ratio color')
parser.add_argument('--data-dir', required=True,
                    help='Directory that contains all labels and images')
parser.add_argument('--output-dir', required=True,
                    help='Directory that outputs processed labels and images')
args = parser.parse_args()

images, _ = images_io.read_images(args.data_dir + '/images')
labels, _ = images_io.read_images(args.data_dir + '/labels')

images = transform.crop(images, crop_size)
labels = transform.crop(labels, crop_size)

if not args.crop_only:
    labels = transform.regularize(labels)
    images, labels = transform.augment(
        images, labels, (args.size, args.size), stride=args.stride)
    images, labels = transform.cutout(
        images, labels, threshold=args.threshold)

output_images_dir = args.output_dir + '/images'
output_labels_dir = args.output_dir + '/labels'

images_io.write_images(output_images_dir, images)
images_io.write_images(output_labels_dir, labels)
