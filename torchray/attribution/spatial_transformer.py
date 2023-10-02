import torch
import torch.nn.functional as F
import torch.optim as optim

# Define a meshgrid for spatial transformer
def compute_grid(H, W):
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
    grid = torch.stack([grid_x, grid_y], dim=2)
    return grid

def get_scaled(x,s):
    grid = compute_grid(x.shape[2], x.shape[3])
    # Scale the grid by s
    grid = grid * s
    # Sample the original image tensor using the scaled grid
    resized_x = F.grid_sample(x, grid.unsqueeze(0), mode='bilinear', align_corners=True, padding_mode='zeros')
    return resized_x
def main():
    # Dummy image tensor (batch_size=1, channels=3, height=64, width=64)
    x = torch.rand((1, 3, 64, 64))

    # Scalar s initialized to 1 (i.e., original size) and supposed to be optimized
    s = torch.nn.Parameter(torch.tensor(1.0))

    optimizer = optim.SGD([s], lr=0.01)

    num_epochs = 100

    for epoch in range(num_epochs):

        # Create a grid
        grid = compute_grid(x.shape[2], x.shape[3])

        # Scale the grid by s
        grid = grid * s

        # Sample the original image tensor using the scaled grid
        resized_x = F.grid_sample(x, grid.unsqueeze(0), mode='bilinear', align_corners=True, padding_mode='zeros')

        # Your objective (loss) computation here
        loss = -resized_x.mean()

        # Gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Scale: {s.item()}, Loss: {loss.item()}")

    # Now s is optimized and you can sample from x at the optimized scale
    optimized_resized_x = F.grid_sample(x, (compute_grid(x.shape[2], x.shape[3]) * s).unsqueeze(0), mode='bilinear', align_corners=True, padding_mode='zeros')
