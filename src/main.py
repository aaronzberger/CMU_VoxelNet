from dataset import get_data_loaders
from voxel_net import VoxelNet
import argparse
import torch
from utils import load_config
from voxel_loss import VoxelLoss
from tqdm import tqdm
import time


def build_model(device):
    start_time = time.time()
    config = load_config()
    net = VoxelNet(device)
    net = net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=config['learning_rate'])

    scheduler = None

    loss_fn = VoxelLoss(config['voxel_loss_params']['alpha'],
                        config['voxel_loss_params']['beta'])

    print('Building model took {:.5f} seconds'.format(time.time() - start_time))

    return net, loss_fn, optimizer, scheduler


def train(device):
    config = load_config()
    net, loss_fn, optimizer, scheduler = build_model(device)

    train_data_loader, _, num_train, _ = get_data_loaders(
        batch_size=config['batch_size'], shuffle=False)

    # INSERT RESUME TRAINING

    end_epoch = config['max_epochs']

    for epoch in range(0, end_epoch):
        epoch_loss = 0
        net.train()

        with tqdm(total=num_train, desc='Epoch %s/%s' % (epoch, end_epoch),
                  unit='pointclouds', leave=True, colour='green') as progress:

            for voxel_features, voxel_coords, pos_equal_one, neg_equal_one, \
                    targets, image, calibs, ids in train_data_loader:

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

                progress.set_postfix(
                    **{'loss': '{:.4f}'.format(abs(loss.item()))})

                epoch_loss += loss

                progress.update(config['batch_size'])

        epoch_loss = epoch_loss / len(train_data_loader)
        print('Epoch {}: Training Loss: {:.5f}'.format(epoch, epoch_loss))

        # INSERT SAVE CHECKPOINT

        if scheduler is not None:
            scheduler.step()

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
