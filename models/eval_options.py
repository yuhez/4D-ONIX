"""
    Test options. A brutal implementation so that many are used from training options.
"""

import os
from datetime import datetime
import argparse


class ParamOptions(object):
    """This class defines options used during test time."""

    def __init__(self):
        super(ParamOptions, self).__init__()

        self.initialized = False
        self.time = datetime.now()
        self.cwd = os.getcwd()

    def initialize(self, parser):
        parser.add_argument(
            "--in_channel",
            type=int,
            default=2,
            help="Number of input channels. Both phase and attenuation channels are used by default.",
        )  # For the current implementation, d_out=in_channel. Remember to change d_out when changing in_channel.
        parser.add_argument(
            "--load_path",
            "-p",
            type=str,
            default="data.npy",
            help="Path to the training file, the customized dataset use npy files",
        )
        parser.add_argument(
            "--n_views",
            "-n",
            type=int,
            default=4,
            help="Number of views used as constraints",
        )
        parser.add_argument(
            "--batch_size",
            "-b",
            type=int,
            default=1,
            help="Number of objects trained in each epoch",
        )
        parser.add_argument(
            "--num_val_views",
            "-v",
            type=int,
            default=2,
            help="Number of validation views. If you want to use all views for the training, leave it 0, and the validation will be generated from a random view",
        )
        parser.add_argument(
            "--val_view_idx",
            type=str,
            nargs="+",
            default="1, 3, 5, 7",
            help="View index for the constraints. Need to match number of views.",
        )
        parser.add_argument(
            "--val_idx",
            type=int,
            default=9,
            help="Index of view angle for the validation. Set to -1 if you want to validate using a random view.",  # todo: Only works for specific senerios. Implement for convenience.
        )
        parser.add_argument(
            "--num_random_rays", type=int, default=1024, help="Number of rays"
        )
        parser.add_argument(
            "--num_encoding_fn_xyz",
            type=int,
            default=10,
            help="Degree of positional encoding for xyz.",
        )
        parser.add_argument(
            "--test_obj_idx",
            type=int,
            default=22,
            help="Index of test object",
        )
        parser.add_argument(
            "--gpu_id",
            type=int,
            default=0,
            help="GPU ID. Currently only one gpu is supported",
        )
        parser.add_argument(
            "--include_input_xyz",
            action="store_true",
            default=True,
            help="Include xyz in the mlp input",
        )
        parser.add_argument(
            "--log_sampling_xyz",
            action="store_true",
            default=True,
            help="Use log sampling or linear sampling. By default log sampling is used.",
        )
        parser.add_argument(
            "--num_coarse",
            type=int,
            default=256,
            help="Number of depth samples per ray for the coarse network",
        )
        parser.add_argument(
            "--lindisp",
            action="store_true",
            help="Sample linearly in disparity space, as opposed to in depth space.",
        )
        parser.add_argument("--z_near", type=float, default=0.5, help="Near bound")
        parser.add_argument("--z_far", type=float, default=1.5, help="Far bound")
        parser.add_argument(
            "--chunksize",
            type=int,
            default=2048,
            help="Used for get_minibatch. For the current implementation this need to be divisible with no remainder by H*W",
        )  # For multi view only works when bigger than (n_views x num_random_rays)
        parser.add_argument(
            "--perturb",
            action="store_true",
            default=False,
            help="Whether or not to perturb the sampled depth values.",
        )
        parser.add_argument(
            "--num_fine",
            type=int,
            default=0,
            help="Number of depth samples per ray for the fine network. We have not tested the fine network, but feel that it would be good to keep this option here.",
        )
        parser.add_argument(
            "--use_encoder",
            action="store_true",
            default=True,
            help="Whether or not to use the encoder. We have not carefully tested the no encoder option.",
        )
        parser.add_argument(
            "--encoder_pretrain",
            action="store_true",
            default=True,
            help="Pretrain the encoder (suggested).",
        )
        parser.add_argument(
            "--encoder_num_layers",
            type=int,
            default=3,
            help="Number of layers in the encoder.",
        )
        parser.add_argument(
            "--use_first_pool",
            action="store_true",
            default=True,
            help="Use the first pooling layer.",
        )
        parser.add_argument(
            "--index_interp", type=str, default="bilinear", help="Type of interpolation"
        )
        parser.add_argument(
            "--index_padding", type=str, default="border", help="Type of padding"
        )
        parser.add_argument(
            "--use_camera_space_pnts",
            action="store_true",
            default=True,
            help="Use camera space points.",
        )
        parser.add_argument(
            "--backbone", type=str, default="resnet34", help="Feature extractor"
        )
        parser.add_argument(
            "--n_blocks", type=int, default=5, help="Number of MLP blocks."
        )
        parser.add_argument(
            "--d_hidden", type=int, default=128, help="Number of hidden layers in MLP."
        )
        parser.add_argument(
            "--d_out", type=int, default=2, help="Number of MLP output layers."
        )
        parser.add_argument(
            "--combine_layer",
            type=int,
            default=3,
            help="Combine after the third layer by average.",
        )
        parser.add_argument(
            "--combine_type",
            type=str,
            default="average",
            help="Feature combination type: average or max",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default="pretrain",
            help="Path to the pretrain run directory",
        )
        parser.add_argument(
            "--load_epoch",
            type=int,
            default=300,
            help="Load epochs if load_pretrain is used",
        )

        parser.add_argument(
            "--use_time",
            action="store_true",
            default=False,
            help="Whether or not to include time",
        )
        parser.add_argument(
            "--eval_H",
            type=int,
            default=None,
            help="If specified, use this as the height for the reconstruction",
        )
        parser.add_argument(
            "--eval_W",
            type=int,
            default=None,
            help="If specified, use this as the width for the reconstruction",
        )
        parser.add_argument(
            "--num_encoding_fn_time",
            type=int,
            default=6,
            help="Degree of positional encoding for time",
        )
        parser.add_argument(
            "--include_input_time",
            action="store_true",
            default=True,
            help="Include time in the mlp input",
        )
        parser.add_argument(
            "--log_sampling_time",
            action="store_true",
            default=True,
            help="Use log sampling or linear sampling. By default log sampling is used.",
        )
        parser.add_argument(
            "--random_proj",
            action="store_true",
            default=False,
            help="Whether or not to use random projection angles in the training. This is mainly used when there are few views available, and the network is supervised by the GAN loss.",
        )

        parser.add_argument(
            "-h",
            "--help",
            action="help",
            default=argparse.SUPPRESS,
            help="Invoke help functions",
        )
        self.initialized = True
        return parser

    def list_view_idx(self, opt, arg):
        d = vars(opt)
        if arg in d.keys():
            d[arg] = [int(s.strip()) for s in d[arg][0].split(",")]

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
            )
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        self.list_view_idx(opt=self.opt, arg="val_view_idx")
        return self.opt
