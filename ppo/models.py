from flax import nn


class CNN(nn.Module):
    def apply(self, x, action_shape):
        x = nn.Conv(x, features=29, kernel_size=(3, 3))
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(x, features=64, kernel_size=(3, 3))
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        a = nn.Dense(x, features=action_shape)
        v = nn.Dense(x, features=1)
        return a, v
