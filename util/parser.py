import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # Dataloader
    parser.add_argument('--name', type=str, default='CelebA-HQ',help='the name of this trainning')
    parser.add_argument('--data_dir', default='datasets/CelebA-HQ', help='path to the image folder')
    parser.add_argument('--img_size', type=int, default=256, help='The size of images for training and validation')
    parser.add_argument('--style_dim',type=int, default=256, help='The size of style code for enconder')
    parser.add_argument('--batch_size', type=int, default=1,help='Batch size for the dataloaders for train and val set')
    parser.add_argument('--num_workers', type=int, default=0,help='Number of CPU cores you want to use for data loading')
    parser.add_argument('--suffix', type=str, default='png',help='the suffix of image file')
    parser.add_argument('--color_file', type=str, default=None,help='the colors in dataset')
    parser.add_argument('--label_nc', type=int, default=1,help='the label channels in dataset')
    parser.add_argument('--random_flip', type=bool, default=False,help='the label channels in dataset')


    # Training arguments
    parser.add_argument('--mode', type=str, default='train',help='the running mode of the model')
    parser.add_argument('--total_epoch', type=int, default=300,help='Number of epochs to run of training')
    parser.add_argument('--lr', type=float, default=1e-4,help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.1,help='gamma for MultiStepLR')
    parser.add_argument('--lm', type=int, default=100,help='lower limit for milestones')
    parser.add_argument('--um', type=int, default=200,help='upper limit for milestones')
    parser.add_argument('--lambda_vgg', type=float, default=10,help='hyperparameters for vggloss')
    parser.add_argument('--lambda_feat', type=float, default=10,help='hyperparameters for feture matching loss')
    parser.add_argument('--lambda_kld', type=float, default=0.01,help='hyperparameters for kld loss')
    parser.add_argument('--use_adain', type=bool, default=True,help='use the spadeadinresblk or not ')
    parser.add_argument('--num_D', type=int, default=3,help='Number of discriminator in multiscale discriminator')
    parser.add_argument('--n_layers_D', type=int, default=3,help='Number of layers per discriminator')
    parser.add_argument('--dataset_mode', type=str, default='LabelDataset',help='the parsing mode of the dataset')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--restart_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--resume_epoch', type=int, default=0,help='Number of epochs for loading checkpoints')
    parser.add_argument('--resume_iter', type=int, default=0,help='Number of iters for loading checkpoints')
    
    #Logging arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',help='Checkpoints directory for saving checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',help='Result directory for saving results')
    parser.add_argument('--log_dir',type=str, default='logs', help='Log directory for saving logs')
    parser.add_argument('--resume_dir', type=str, default=None,help='Resume directory for loading checkpoints')
    parser.add_argument('--display_every_iter',type=int, default=100, help='save per epochs')
    parser.add_argument('--save_every_iter',type=int, default=1000,help='save per iter')
    parser.add_argument('--save_every_epoch',type=int, default=20, help='save per epochs')
    
    
    return parser