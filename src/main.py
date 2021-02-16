from dataset import get_data_loader
from voxel_net import VoxelNet
import argparse
import torch
from utils import load_config, draw_boxes, get_model_path
from viz_3d import save_center_batch
from voxel_loss import VoxelLoss
from tqdm import tqdm
import numpy as np
import csv


def build_model(device):
    config = load_config()
    net = VoxelNet(device)
    net = net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=config['learning_rate'])

    scheduler = None

    loss_fn = VoxelLoss(config['voxel_loss_params']['alpha'],
                        config['voxel_loss_params']['beta'])

    return net, loss_fn, optimizer, scheduler


def train(device):
    config = load_config()
    net, loss_fn, optimizer, scheduler = build_model(device)

    train_data_loader, num_train = get_data_loader()

    # Resume training if needed
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
        with tqdm(total=end_epoch, desc='Training', unit='epochs', leave=True,
                  colour='magenta', position=0, initial=start_epoch - 1) \
                as epoch_tracker:

            if epoch % config['viz_every'] == 0:
                indices = np.random.choice(
                    np.arange(len(train_data_loader) // config['batch_size']),
                    replace=False, size=(config['num_viz']))
            else:
                indices = []

            epoch_loss = 0
            epoch_losses = []
            net.train()

            with tqdm(total=num_train, desc='Epoch %s/%s' % (epoch, end_epoch),
                      unit='pointclouds', leave=True, colour='green',
                      position=1) as progress:

                for voxel_features, voxel_coords, pos_equal_one, \
                    neg_equal_one, targets, lidar, image, calibs, ids \
                        in train_data_loader:

                    optimizer.zero_grad()

                    # Forward Prop
                    voxel_features = torch.Tensor(voxel_features).to(device)
                    voxel_coords = torch.Tensor(voxel_coords).to(device)
                    prob_score_map, reg_map = net(voxel_features, voxel_coords)

                    pos_equal_one = torch.Tensor(pos_equal_one).to(device)
                    neg_equal_one = torch.Tensor(neg_equal_one).to(device)
                    targets = torch.Tensor(targets).to(device)

                    # Loss
                    loss, conf_loss, reg_loss = loss_fn(
                        prob_score_map, reg_map,
                        pos_equal_one, neg_equal_one, targets)

                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.item())

                    with torch.no_grad():
                        if progress.n // config['batch_size'] in indices:
                            bounding_boxes, scores = draw_boxes(
                                reg_map, prob_score_map)
                            save_center_batch(
                                batch_boxes=bounding_boxes.cpu().numpy(),
                                batch_lidar=lidar, epoch=epoch, ids=ids)

                    progress.set_postfix(
                        **{'loss': '{:.4f}'.format(abs(loss.item()))})

                    epoch_loss += loss

                    progress.update(config['batch_size'])

            epoch_tracker.update()

            all_losses.append(epoch_losses)

            with open('/home/aaron/losses.csv', 'w') as loss_writer:
                csv_writer = csv.writer(loss_writer)
                for row in all_losses:
                    csv_writer.writerow(row)

            epoch_loss = epoch_loss / len(train_data_loader)
            print('Epoch {}: Training Loss: {:.5f}'.format(
                epoch, epoch_loss))

            # Save model state
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
    if args.mode == 'test':
        raise NotImplementedError()
