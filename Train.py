import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training weather prediction model')
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--cuda', type=str, default=-1)
    parser.add_argument('--load_model', action='store_true', default=False)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    from modules.full_pipeline_multiple_steps import main
    main(args.config_file, args.load_model)