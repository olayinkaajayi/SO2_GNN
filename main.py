import os, yaml, argparse
from src.generator import Generator
from src.processor import Processor

# Thanks to Song et al for the released code on Github (https://github.com/zyxjtu/EfficientGCNv1)
def main():
    # Loading parameters
    parser = init_parser()
    args = parser.parse_args()
    args = update_parameters(parser, args)  # cmd > yaml > default

    # Processing
    if args.generate_data:
        g = Generator(args)
        g.start()

    elif args.transform_data:
        # This would add translation, alignment (and rotation) to the skeletons.
        # It would also provide us with the train-test split: x-view, x-sub
        g = Generator(args)
        g.start()
        print("\n****Tranformation Complete!****")

    elif args.extract:
        p = Processor(args)
        p.extract()
    
    elif args.see_model_vals:
        p = Processor(args)
        p.see_model_vals()

    else:
        p = Processor(args)
        p.start()


def init_parser():
    parser = argparse.ArgumentParser(description='Paranomic GCN for skeleton-based GAR')

    # Setting
    parser.add_argument('--config', '-c', default='', help='path to the config file', required=True)
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')
    parser.add_argument('--pretrained_path', '-pp', type=str, default='', help='Path to pretrained models')
    parser.add_argument('--work_dir', '-w', type=str, default='', help='Work dir')
    parser.add_argument('--experiment_name', '-ep', type=str, default='', help='experiment name')

    # Processing
    parser.add_argument('--debug', '-db', default=False, action='store_true', help='Debug')
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('--see_model_vals', '-sm', default=False, action='store_true', help='check model vales from checkpoint') #######
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')
    parser.add_argument('--generate_data', '-gd', default=False, action='store_true', help='Generate skeleton data')
    parser.add_argument('--transform_data', '-td', default=False, action='store_true', help='Transform skeleton data')

    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='', help='Select dataset')
    parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')
    parser.add_argument('--alignment', '-al', type=str, default='aligned', help='decide if aligned or unaligned')
    parser.add_argument('--case', type=str, default='CV', help='decide if CV or CS')

    # Model
    parser.add_argument('--model_type', '-mt', type=str, default='', help='Args for creating model')
    parser.add_argument('--model_args', default=dict(), help='Args for creating model')
    
    # Optimizer
    parser.add_argument('--optimizer', '-o', type=str, default='', help='Initial optimizer')
    parser.add_argument('--optimizer_args', default=dict(), help='Args for optimizer')

    # LR_Scheduler
    parser.add_argument('--lr_scheduler', '-ls', type=str, default='', help='Initial learning rate scheduler')
    parser.add_argument('--scheduler_args', default=dict(), help='Args for scheduler')

    return parser


def update_parameters(parser, args):
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this file in \'config\' folder: {}!'.format(args.config))
    return parser.parse_args()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
