
import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import base_dir
from conversions import output_to_boxes
from dataset import get_data_loader
from utils import load_config, get_model_path, mkdir_p
from viz_3d import save_viz_batch
from voxel_loss import VoxelLoss
from voxel_net import VoxelNet


def build_model(device):
    '''
    Setup the model, loss function, optimizer, and scheduler

    Parameters:
        device (torch.device): device to put everything on
    '''
    config = load_config()
    net = VoxelNet(device)
    net = net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=config['learning_rate'])

    scheduler = None

    loss_fn = VoxelLoss(config['voxel_loss_params']['alpha'],
                        config['voxel_loss_params']['beta'])

    return net, loss_fn, optimizer, scheduler


def train(device):
    '''
    Train the network

    Parameters:
        device (torch.device): device to put everything on
    '''
    # Setup the Tensorboard logger
    mkdir_p(os.path.join(base_dir, 'log'))
    writer = SummaryWriter(log_dir=os.path.join(base_dir, 'log'))
    writer_counter = 0

    config = load_config()

    print('''\nLoaded hyperparameters:
    Learning Rate:   %s
    Batch size:      %s
    Epochs:          %s
    Device:          %s
    ''' % (config['learning_rate'], config['batch_size'], config['max_epochs'],
           'CPU' if device == torch.device('cpu') else 'GPU'))

    net, loss_fn, optimizer, scheduler = build_model(device)

    # Get the data from dataset.py
    train_data_loader, num_train = get_data_loader()

    # Resume training from a certain epoch if specified in config json file
    if config['resume_from'] != 0:
        start_epoch = config['resume_from']
        saved_ckpt_path = get_model_path(start_epoch)

        net.load_state_dict(
            torch.load(saved_ckpt_path, map_location=device))

        print('Loaded model dict from {}'.format(saved_ckpt_path))
    else:
        start_epoch = 1

    end_epoch = config['max_epochs']
    all_losses = []

    for epoch in range(start_epoch, end_epoch):
        # Specify indices in this epoch that will be visualized, as specified
        # in the config json file params
        if epoch % config['viz_every'] == 0 and not config['viz_none']:
            indices = np.random.choice(
                np.arange(len(train_data_loader) // config['batch_size']),
                replace=False, size=(config['num_viz']))
        else:
            indices = []

        epoch_loss = 0
        epoch_losses = []
        net.train()

        # Progress bar for each epoch
        with tqdm(total=num_train, desc='Epoch %s/%s' % (epoch, end_epoch),
                  unit='pointclouds', leave=False, colour='green') as progress:

            for voxel_features, voxel_coords, pos_equal_one, \
                neg_equal_one, targets, gt_bounding_boxes, lidar, image, \
                    calibs, ids in train_data_loader:

                optimizer.zero_grad()

                voxel_features = torch.Tensor(voxel_features).to(device)
                voxel_coords = torch.Tensor(voxel_coords).to(device)

                # Pass through the network
                prob_score_map, reg_map = net(voxel_features, voxel_coords)

                pos_equal_one = torch.Tensor(pos_equal_one).to(device)
                neg_equal_one = torch.Tensor(neg_equal_one).to(device)
                targets = torch.Tensor(targets).to(device)

                # Calculate loss
                loss, conf_loss, reg_loss = loss_fn(
                    prob_score_map, reg_map,
                    pos_equal_one, neg_equal_one, targets)

                loss.backward()
                optimizer.step()

                # Add loss info to the Tensorboard logger
                writer.add_scalars(
                    main_tag="training",
                    tag_scalar_dict={
                        "loss": loss.item(),
                        "conf_loss": conf_loss.item(),
                        "reg_loss": reg_loss.item(),
                    }, global_step=writer_counter)
                writer_counter += 1

                # Visualize this image
                with torch.no_grad():
                    if progress.n // config['batch_size'] in indices:
                        # [BS, C, W, H] -> [BS, W, H, C]
                        prob_score_map = prob_score_map.permute(
                            0, 2, 3, 1).contiguous()
                        reg_map = reg_map.permute(
                            0, 2, 3, 1).contiguous()

                        # Convert VoxelNet output to bounding boxes
                        boxes_corner, _ = output_to_boxes(
                            prob_score_map, reg_map)

                        if boxes_corner is not None:
                            # Save the bounding boxes to a file (viz folder)
                            save_viz_batch(
                                pointcloud=lidar, boxes=boxes_corner,
                                gt_boxes=gt_bounding_boxes,
                                epoch=epoch, ids=ids)

                # Update progress bar
                progress.set_postfix(
                    **{'loss': '{:.4f}'.format(abs(loss.item()))})
                progress.update(config['batch_size'])

                epoch_losses.append(loss.item())
                epoch_loss += loss

        all_losses.append(epoch_losses)

        # Write a new csv file containing all epoch losses
        with open('/home/aaron/losses.csv', 'w') as loss_writer:
            csv_writer = csv.writer(loss_writer)
            for row in all_losses:
                csv_writer.writerow(row)

        epoch_loss = epoch_loss / len(train_data_loader)
        print('Epoch {}: Training Loss: {:.5f}'.format(
            epoch, epoch_loss))

        # Save model dict as specified by params in config json
        if epoch == end_epoch or \
                epoch % config['save_every'] == 0:
            torch.save(net.state_dict(), get_model_path(name=epoch))

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VoxelNet')
    parser.add_argument(
        'mode', choices=['train', 'test'], help='mode for the model')
    args = parser.parse_args()

    # Choose a device for the model
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if args.mode == 'train':
        train(device)
    else:
        raise NotImplementedError()
