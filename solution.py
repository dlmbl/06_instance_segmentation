# %% [markdown]
# # Exercise 05: Instance Segmentation
#
# So far, we were only interested in `semantic` classes, e.g. foreground / background etc.
# But in many cases we not only want to know if a certain pixel belongs to a specific class, but also to which unique object (i.e. the task of `instance segmentation`).
#
# For isolated objects, this is trivial, all connected foreground pixels form one instance, yet often instances are very close together or even overlapping. Thus we need to think a bit more how to formulate the targets / loss of our network.
#
# Furthermore, in instance segmentation the specific value of each label is arbitrary. Here we label each cell with a number and assign a color to each number giving us a segmentation mask. `Mask 1` and `Mask 2` are equivalently good segmentations even though the specific label of each cell is arbitrary.
#
# | Image | Mask 1| Mask 2|
# | :-: | :-: | :-: |
# | ![image](static/figure1/01_instance_image.png) | ![mask1](static/figure1/02_instance_teaser.png) | ![mask2](static/figure1/03_instance_teaser.png) |
#
# Once again: THE SPECIFIC VALUES OF THE LABELS ARE ARBITRARY
#
# This means that the model will not be able to learn, if tasked to predict the labels directly.
#
# Therefore we split the task of instance segmentation in two and introduce an intermediate target which must be:
#   1) learnable
#   2) post-processable into an instance segmentation
#
# In this exercise we will go over two common intermediate targets (signed distance transform and affinities),
# as well as the necessary pre and post-processing for obtaining the final segmentations.
#
# At the end of the exercise we will also compare to a pre-trained cellpose model.

# %% [markdown]
# <div class="alert alert-block alert-danger">
# <b>Conda Kernel</b>: Please use the kernel `04-instance-segmentation` for this exercise
# </div>

# %% [markdown]
# ## Section 0: Imports and Setup

# %%
# Set start method for MacOS
import multiprocessing

multiprocessing.set_start_method("fork", force=True)

# %% [markdown]
# ## Import Packages
# %%
from matplotlib.colors import ListedColormap
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2
from scipy.ndimage import distance_transform_edt, map_coordinates
from local import train, NucleiDataset, plot_two, plot_three, plot_four
from dlmbl_unet import UNet
from tqdm import tqdm
import tifffile
import mwatershed as mws

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects


# %%
# Set some variables that are specific to the hardware being run on
# this should be optimized for the compute nodes once available.
device = "cpu"  # 'cuda', 'cpu', 'mps'
NUM_THREADS = 0
NUM_EPOCHS = 20
# make sure gpu is available. Please call a TA if this cell fails
# assert torch.cuda.is_available()

# %%
# Create a custom label color map for showing instances
np.random.seed(1)
colors = [[0, 0, 0]] + [list(np.random.choice(range(256), size=3)) for _ in range(254)]
label_cmap = ListedColormap(colors)

# %% [markdown]
# ## Section 1: Signed Distance Transform (SDT)
#
# First we will use the signed distance transform as an intermediate learning objective
#
# <i>What is the signed distance transform?</i>
# <br>  - Signed Distance Transform indicates the distance from each specific pixel to the boundary of objects.
# <br>  - It is positive for pixels inside objects and negative for pixels outside objects (i.e. in the background).
# <br>  - Remember that deep learning models work best with normalized values, therefore it is important the scale the distance
#            transform. For simplicity things are often scaled between -1 and 1.
# <br>  - As an example, here, you see the SDT (right) of the target mask (middle), below.

# %% [markdown]
# ![image](static/figure2/04_instance_sdt.png)
#

# %%

def compute_sdt(labels: np.ndarray, scale: int = 5):
    """Function to compute a signed distance transform."""
    dims = len(labels.shape)
    # Create a placeholder array of infinite distances
    distances = np.ones(labels.shape, dtype=np.float32) * np.inf
    for axis in range(dims):
        # Here we compute the boundaries by shifting the labels and comparing to the original
        # This can be visualized in 1D as:
        # a a a b b c c c
        #   a a a b b c c c
        #   1 1 0 1 0 1 1
        # Applying a half pixel shift makes the result more obvious:
        # a a a b b c c c
        #  1 1 0 1 0 1 1
        bounds = (
            labels[*[slice(None) if a != axis else slice(1, None) for a in range(dims)]]
            == labels[
                *[slice(None) if a != axis else slice(None, -1) for a in range(dims)]
            ]
        )
        # pad to account for the lost pixel
        bounds = np.pad(
            bounds,
            [(1, 1) if a == axis else (0, 0) for a in range(dims)],
            mode="constant",
            constant_values=1,
        )
        # compute distances on the boundary mask
        axis_distances = distance_transform_edt(bounds)

        # compute the coordinates of each original pixel relative to the boundary mask and distance transform.
        # Its just a half pixel shift in the axis we computed boundaries for.
        coordinates = np.meshgrid(
            *[
                range(axis_distances.shape[a])
                if a != axis
                else np.linspace(0.5, axis_distances.shape[a] - 1.5, labels.shape[a])
                for a in range(dims)
            ],
            indexing="ij",
        )
        coordinates = np.stack(coordinates)

        # Interpolate the distances to the original pixel coordinates
        sampled = map_coordinates(
            axis_distances,
            coordinates=coordinates,
            order=3,
        )

        # Update the distances with the minimum distance to a boundary in this axis
        distances = np.minimum(distances, sampled)

    # Normalize the distances to be between -1 and 1
    distances = np.tanh(distances / scale)

    # Invert the distances for pixels in the background
    distances[labels == 0] *= -1
    return distances


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.1</b>: Explain the `compute_sdt` from the cell above.
# </div>


# %% [markdown] tags=["task"]
# 1. _Why do we need to loop over dimensions? Couldn't we do all at once?_
#
# 2. _What is the purpose of the pad?_
#
# 3. _What does meshgrid do?_
#
# 4. _Why do we use `map_coordinates`?_
#
# 5. _ bonux question: Is the pad sufficient to give us accurate distances at the edge of our image?_

# %% [markdown] tags=["solution"]
# 1. _Why do we need to loop over dimensions? Couldn't we do all at once?_
# To get the distance to boundaries in each axis. Regardless of the shift we choose, we will always miss boundaries that line up perfectly with the offset. (shifting by (1, 1) will miss diagonal boundaries).
#
# 2. _What is the purpose of the pad?_
# We lose a pixel when we compute the boundaries so we need to pad to cover the whole input image.
#
# 3. _What does meshgrid do?_
# It computes the index coordinate of every voxel. Offset by half on the dimension along which we computed boundaries because the boundaries sit half way between the voxels on either side of the boundary
#
# 4. _Why do we use `map_coordinates`?_
# Boundaries are defined between pixels, not on individual pixels. So the distance from a pixel on a boundary to the boundary should be half of a pixel. Map Coordinates lets us get this interpolation<
#
# 5. _ bonux question: Is the pad sufficient to give us accurate distances at the edge of our image?_
# Kind of. If you assume this is the full image and no data exists outside the provided region, then yes. But if you have a larger image, then you cannot know the distance to the nearest out of view object. It might be visible given one more pixel, or there could never be another object.
# Depending on how you train, you may need to take this into account.

# %% [markdown]
# Below is a quick visualization of the signed distance transform (SDT).
# <br> Note that the output of the signed distance transform is not binary, a significant difference from semantic segmentation
# %%
# Visualize the signed distance transform using the function you wrote above.

root_dir = "tissuenet_data/train"  # the directory with all the training samples
samples = os.listdir(root_dir)
idx = np.random.randint(len(samples) // 3)  # take a random sample.
img = tifffile.imread(os.path.join(root_dir, f"img_{idx}.tif"))  # get the image
label = tifffile.imread(
    os.path.join(root_dir, f"img_{idx}_cyto_masks.tif")
)  # get the image
sdt = compute_sdt(label)
plot_three(img, label, sdt, label="SDT", label_cmap=label_cmap)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.2</b>: Explain the scale parameter.
# </div>

# %% [markdown] tas=["task"]
# <b>Questions</b>:
# 1. _Why do we need to normalize the distances between -1 and 1?_
#
# 2. _What is the effect of changing the scale value? What do you think is a good default value?_
#

# %% [markdown] tags=["solution"]
# <b>Questions</b>:
# 1. _Why do we need to normalize the distances between -1 and 1?_
#   If the closest object to a pixel is outside the receptive field, the model cannot know whether the distance is 100 or 100_000. Squeezing large distances down to 1 or -1 makes the answer less ambiguous.

# 2. _What is the effect of changing the scale value? What do you think is a good default value?_
#   Increasing the scale is equivalent to having a wider boundary region. 5 seems reasonable.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.3</b>: <br>
#     Modify the `SDTDataset` class below to produce the paired raw and SDT images.<br>
#   1. Fill in the `create_sdt_target` method to return an SDT output rather than a label mask.<br>
#       - Ensure that all final outputs are of torch tensor type, and are converted to float.
#   2. Instantiate the dataset with a RandomCrop of size 256 and visualize the output to confirm that the SDT is correct.
# </div>


# %% tags=["task"]
class SDTDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks."""

    def __init__(self, root_dir, transform=None, img_transform=None, return_mask=False):
        self.root_dir = root_dir  # the directory with all the training samples
        self.num_samples = len(os.listdir(self.root_dir)) // 3  # list the samples
        self.return_mask = return_mask
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )
        self.from_np = v2.Lambda(lambda x: torch.from_numpy(x))

        self.loaded_imgs = [None] * self.num_samples
        self.loaded_masks = [None] * self.num_samples
        for sample_ind in tqdm(range(self.num_samples)):
            img_path = os.path.join(self.root_dir, f"img_{sample_ind}.tif")
            image = self.from_np(tifffile.imread(img_path))
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            mask_path = os.path.join(self.root_dir, f"img_{sample_ind}_cyto_masks.tif")
            mask = self.from_np(tifffile.imread(mask_path))
            self.loaded_masks[sample_ind] = mask

    # get the total number of samples
    def __len__(self):
        return self.num_samples

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # We'll be using the Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        # use the compute_sdt function to get the sdt
        sdt = self.create_sdt_target(mask)
        assert isinstance(sdt, torch.Tensor)
        assert sdt.dtype == torch.float32
        assert sdt.shape == mask.shape
        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.return_mask is True:
            return image, mask.unsqueeze(0), sdt.unsqueeze(0)
        else:
            return image, sdt.unsqueeze(0)

    def create_sdt_target(self, mask):
        ...
        return ...

# %% tags=["solution"]
class SDTDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks."""

    def __init__(self, root_dir, transform=None, img_transform=None, return_mask=False):
        self.root_dir = root_dir  # the directory with all the training samples
        self.num_samples = len(os.listdir(self.root_dir)) // 3  # list the samples
        self.return_mask = return_mask
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )
        self.from_np = v2.Lambda(lambda x: torch.from_numpy(x))

        self.loaded_imgs = [None] * self.num_samples
        self.loaded_masks = [None] * self.num_samples
        for sample_ind in tqdm(range(self.num_samples), desc="Reaqding Images"):
            img_path = os.path.join(self.root_dir, f"img_{sample_ind}.tif")
            image = self.from_np(tifffile.imread(img_path))
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            mask_path = os.path.join(self.root_dir, f"img_{sample_ind}_cyto_masks.tif")
            mask = self.from_np(tifffile.imread(mask_path))
            self.loaded_masks[sample_ind] = mask

    # get the total number of samples
    def __len__(self):
        return self.num_samples

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # We'll be using the Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        sdt = self.create_sdt_target(mask)
        assert sdt.shape == mask.shape
        assert isinstance(sdt, torch.Tensor)
        assert sdt.dtype == torch.float32
        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.return_mask is True:
            return image, mask.unsqueeze(0), sdt.unsqueeze(0)
        else:
            return image, sdt.unsqueeze(0)

    def create_sdt_target(self, mask):
        sdt_target_array = compute_sdt(mask.numpy())
        sdt_target = self.from_np(sdt_target_array)
        return sdt_target.float()

#%% tags=["task"]
# Create a dataset using a RandomCrop of size 256 (see torchvision.transforms.v2 imported as v2)
# documentation here: https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended
# Visualize the output to confirm your dataset is working.

train_data = SDTDataset("tissuenet_data/train", ...)
img, sdt = train_data[10]  # get the image and the distance transform
# We use the `plot_two` function (imported in the first cell) to verify that our
# dataset solution is correct. The output should show 2 images: the raw image and
# the corresponding SDT.
plot_two(img, sdt[0], label="SDT")

#%% tags=["solution"]
# Create a dataset using a RandomCrop of size 256 (see torchvision.transforms.v2 imported as v2)
# documentation here: https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended
# Visualize the output to confirm your dataset is working.

train_data = SDTDataset("tissuenet_data/train", v2.RandomCrop(256))
img, sdt = train_data[10]  # get the image and the distance transform
# We use the `plot_two` function (imported in the first cell) to verify that our
# dataset solution is correct. The output should show 2 images: the raw image and
# the corresponding SDT.
plot_two(img, sdt[0], label="SDT")

# %% [markdown] tags=["task"]
# <div class="alert alert-block alert-info">
# <b>Task 1.4</b>: Understanding the dataloader.
# Our dataloader has some features that are not straightforward to understand or justify, and this is a good point
# to discuss them.
# 
# 1. _What are we doing with the `seed` variabel and why? Can you predict what will go wrong when you delete the `seed` code and rerun the previous cells visualization?_
# 
# 2. _What is the purpose of the `loaded_imgs` and `loaded_masks` lists?_
# </div>

# %% [markdown] tags=["solution"]
# <div class="alert alert-block alert-info">
# <b>Task 1.4</b>: Understanding the dataloader.
# Our dataloader has some features that are not straightforward to understand or justify, and this is a good point
# to discuss them.
# 
# 1. _What are we doing with the `seed` variabel and why? Can you predict what will go wrong when you delete the `seed` code and rerun the previous cells visualization?_
# The seed variable is used to ensure that the same random transform is applied to the image and mask. If we don't use the seed, the image and mask will be transformed differently, leading to misaligned data.
#
# 2. _What is the purpose of the `loaded_imgs` and `loaded_masks` lists?_
# We load the images and masks into memory to avoid reading them from disk every time we access the dataset. This speeds up the training process. GPUs are very fast so
# we often need to put a lot of thought into how to provide data to them fast enough.
#
# </div>


# %% [markdown]
# Next, we will create a training dataset and data loader.
# 
# %%
# TODO: You don't have to add extra augmentations, training will work without.
# But feel free to experiment here if you want to come back and try to get better results if you have time.
train_data = SDTDataset("tissuenet_data/train", v2.RandomCrop(256))
train_loader = DataLoader(
    train_data, batch_size=5, shuffle=True, num_workers=NUM_THREADS
)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.5</b>: Train the U-Net.
# </div>
# %% [markdown]
# In this task, initialize the UNet, specify a loss function, learning rate, and optimizer, and train the model.<br>
# <br> For simplicity we will use a pre-made training function imported from `local.py`. <br>
# <u>Hints</u>:<br>
#   - Loss function - [torch losses](https://pytorch.org/docs/stable/nn.html#loss-functions)
#   - Optimizer - [torch optimizers](https://pytorch.org/docs/stable/optim.html)
#   - Final Activation - there are a few options (only one is the best)
#       - [sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
#       - [tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh)
#       - [relu](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)

# %% tags=["task"]
# If you manage to get a loss close to 0.1, you are doing pretty well and can probably move on
unet = ...

learning_rate = ...
loss = ...
optimizer = ...

for epoch in range(NUM_EPOCHS):
    train(
        model=...,
        loader=...,
        optimizer=...,
        loss_function=...,
        epoch=...,
        log_interval=2,
        device=device,
    )

# %% tags=["solution"]
unet = UNet(
    depth=3,
    in_channels=2,
    out_channels=1,
    final_activation=torch.nn.Tanh(),
    num_fmaps=16,
    fmap_inc_factor=3,
    downsample_factor=2,
    padding="same",
)

learning_rate = 1e-4
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

for epoch in range(NUM_EPOCHS):
    train(
        unet,
        train_loader,
        optimizer,
        loss,
        epoch,
        log_interval=2,
        device=device,
    )

# %% [markdown]
# Now, let's apply our trained model and visualize some random samples. <br>
# First, we create a validation dataset. <br> Next, we sample a random image from the dataset and input into the model.

# %%
val_data = SDTDataset("tissuenet_data/test")
unet.eval()
idx = np.random.randint(len(val_data))  # take a random sample.
image, sdt = val_data[idx]  # get the image and the nuclei masks.
image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))
image = np.squeeze(image.cpu())
sdt = np.squeeze(sdt.cpu().numpy())
pred = np.squeeze(pred.cpu().detach().numpy())
plot_three(image, sdt, pred)


# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 1 </h2>
#
# At this point we have a model that does what we told it too, but do not yet have a segmentation. <br>
# In the next section, we will perform some post-processing and obtain segmentations from our predictions.
# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 2: Post-Processing
# - See here for a nice overview: [open-cv-image watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html), although the specifics of our code will be slightly different
# - Given the distance transform (the output of our model), we first need to find the local maxima that will be used as seed points
# - The watershed algorithm then expands each seed out in a local "basin" until the segments touch or the boundary of the object is hit.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 2.1</b>: write a function to find the local maxima of the distance transform
# </div>

# %% [markdown]
# <u>Hint</u>: Look at the imports. <br>
# <u>Hint</u>: It is possible to write this function by only adding 2 lines.

# %% tags=["task"]
from scipy.ndimage import label, maximum_filter


def find_local_maxima(distance_transform, min_dist_between_points):
    # Hint: Use `maximum_filter` to perform a maximum filter convolution on the distance_transform

    ...
    seeds, number_of_seeds = ...

    return seeds, number_of_seeds


# %% tags=["solution"]
from scipy.ndimage import label, maximum_filter


def find_local_maxima(distance_transform, min_dist_between_points):
    # Use `maximum_filter` to perform a maximum filter convolution on the distance_transform
    max_filtered = maximum_filter(distance_transform, min_dist_between_points)
    maxima = max_filtered == distance_transform
    # Uniquely label the local maxima
    seeds, number_of_seeds = label(maxima)

    return seeds, number_of_seeds


# %%
# test your function.
from local import test_maximum

test_maximum(find_local_maxima)

# %% [markdown]
# We now use this function to find the seeds for the watershed.
# %%
from skimage.segmentation import watershed


def watershed_from_boundary_distance(
    boundary_distances: np.ndarray,
    inner_mask: np.ndarray,
    id_offset: float = 0,
    min_seed_distance: int = 10,
):
    """Function to compute a watershed from boundary distances."""

    seeds, n = find_local_maxima(boundary_distances, min_seed_distance)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    # calculate our segmentation
    segmentation = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=inner_mask
    )

    return segmentation


def get_inner_mask(pred, threshold):
    inner_mask = pred > threshold
    return inner_mask


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 2.2</b>: <br> Use the model to generate a predicted SDT and then use the watershed function we defined above to get post-process into a segmentation
# </div>

# %% tags=["task"]
idx = np.random.randint(len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks

# get the model prediction
# Hint: make sure set the model to evaluation
# Hint: check the dims of the image, remember they should be [batch, channels, x, y]
# Hint: remember to move model outputs to the cpu and check their dimensions (as you did in task 1.4 visualization)
unet.eval()

# remember to move the image to the device
pred = ...

# turn image, mask, and pred into plain numpy arrays

# Choose a threshold value to use to get the boundary mask.
# Feel free to play around with the threshold.
# hint: If you're struggling to find a good threshold, you can use the `threshold_otsu` function

threshold = ...

# Get inner mask
inner_mask = get_inner_mask(pred, threshold=threshold)

# Get the segmentation
seg = watershed_from_boundary_distance(pred, inner_mask, min_seed_distance=20)

# %% tags=["solution"]
idx = np.random.randint(len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks

# get the model prediction
# Hint: make sure set the model to evaluation
# Hint: check the dims of the image, remember they should be [batch, channels, x, y]
# Hint: remember to move model outputs to the cpu and check their dimensions (as you did in task 1.4 visualization)
unet.eval()

image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))

image = np.squeeze(image.cpu())
mask = np.squeeze(mask.cpu().numpy())
pred = np.squeeze(pred.cpu().detach().numpy())

# Choose a threshold value to use to get the boundary mask.
# Feel free to play around with the threshold.
threshold = threshold_otsu(pred)
print(f"Foreground threshold is {threshold:.3f}")

# Get inner mask
inner_mask = get_inner_mask(pred, threshold=threshold)

# Get the segmentation
seg = watershed_from_boundary_distance(pred, inner_mask, min_seed_distance=20)

# %%
# Visualize the results

plot_four(image[1], mask, pred, seg, label="Target", cmap=label_cmap)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 2.3</b>: <br> Min Seed Distance
# </div>

# %% [markdown] tags=["task"]
# Questions:
# 1. What is the effect of the `min_seed_distance` parameter in watershed?
#       - Experiment with different values.

# %% [markdown] tags=["solution"]
# Questions:
# 1. What is the effect of the `min_seed_distance` parameter in watershed?
#       - Experiment with different values.
#
# The `min_seed_distance` parameter is used to filter out local maxima that are too close to each other. This can be useful to prevent oversegmentation. If the value is too high, you may miss some local maxima, leading to undersegmentation. If the value is too low, you may get too many local maxima, leading to oversegmentation.

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 2 </h2>

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 3: Evaluation
# Many different evaluation metrics exist, and which one you should use is dependant on the specifics of the data.
#
# [This website](https://metrics-reloaded.dkfz.de/problem-category-selection) has a good summary of different options.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 3.1</b>: Pick the best metric to use
# </div>
# %% [markdown] tags=["task"]
# Which of the following should we use for our dataset?:
#   1) [IoU](https://metrics-reloaded.dkfz.de/metric?id=intersection_over_union)
#   2) [Accuracy](https://metrics-reloaded.dkfz.de/metric?id=accuracy)
#   3) [Sensitivity](https://metrics-reloaded.dkfz.de/metric?id=sensitivity) and [Specificity](https://metrics-reloaded.dkfz.de/metric?id=specificity@target_value)
#

# %% [markdown] tags=["solution"]
# Which of the following should we use for our dataset?:
#   1) [IoU](https://metrics-reloaded.dkfz.de/metric?id=intersection_over_union)
#   2) [Accuracy](https://metrics-reloaded.dkfz.de/metric?id=accuracy)
#   3) [Sensitivity](https://metrics-reloaded.dkfz.de/metric?id=sensitivity) and [Specificity](https://metrics-reloaded.dkfz.de/metric?id=specificity@target_value)
#
# We will use Accuracy, Precision, and Recall as our evaluation metrics. IoU is also a good metric to use, but it is more commonly used for semantic segmentation tasks.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 3.2</b>: <br> Evaluate metrics for the validation dataset. Fill in the blanks
# </div>

# %% tags=["task"]
from local import evaluate

# Need to re-initialize the dataloader to return masks in addition to SDTs.
val_dataset = SDTDataset("tissuenet_data/test", return_mask=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=NUM_THREADS
)
unet.eval()

(
    precision_list,
    recall_list,
    accuracy_list,
) = (
    [],
    [],
    [],
)
for idx, (image, mask, sdt) in enumerate(tqdm(val_dataloader)):
    image = image.to(device)
    pred = unet(image)

    image = np.squeeze(image.cpu())
    gt_labels = np.squeeze(mask.cpu().numpy())
    pred = np.squeeze(pred.cpu().detach().numpy())

    # feel free to try different thresholds
    thresh = ...

    # get boundary mask
    inner_mask = ...
    pred_labels = ...
    precision, recall, accuracy = evaluate(gt_labels, pred_labels)
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)

print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean Accuracy is {np.mean(accuracy_list):.3f}")

# %% tags=["solution"]
from local import evaluate

# Need to re-initialize the dataloader to return masks in addition to SDTs.
val_dataset = SDTDataset("tissuenet_data/test", return_mask=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=NUM_THREADS
)
unet.eval()

(
    precision_list,
    recall_list,
    accuracy_list,
) = (
    [],
    [],
    [],
)
for idx, (image, mask, sdt) in enumerate(tqdm(val_dataloader)):
    image = image.to(device)
    pred = unet(image)

    image = np.squeeze(image.cpu())
    gt_labels = np.squeeze(mask.cpu().numpy())
    pred = np.squeeze(pred.cpu().detach().numpy())

    # feel free to try different thresholds
    thresh = threshold_otsu(pred)

    # get boundary mask
    inner_mask = get_inner_mask(pred, threshold=thresh)

    pred_labels = watershed_from_boundary_distance(
        pred, inner_mask, id_offset=0, min_seed_distance=20
    )
    precision, recall, accuracy = evaluate(gt_labels, pred_labels)
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)

print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean Accuracy is {np.mean(accuracy_list):.3f}")


# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 4: Affinities
# %% [markdown]
# <i>What are affinities? </i><br>
# Here we consider not just the pixel but also its direct neighbors.
# <br> Imagine there is an edge between two pixels if they are in the same class and no edge if not.
# <br> If we then take all pixels that are directly and indirectly connected by edges, we get an instance.
# <br> Essentially, we label edges between neighboring pixels as “connected” or “cut”, rather than labeling the pixels themselves. <br>
# Here,  we show the (affinity in x + affinity in y) in the bottom right image.

# %% [markdown]
# ![image](static/figure3/instance_affinity.png)

# %% [markdown]
# Similar to the pipeline used for SDTs, we first need to modify the dataset to produce affinities.

# %%
# create a new dataset for affinities
from local import compute_affinities


class AffinityDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(
        self,
        root_dir,
        transform=None,
        img_transform=None,
        return_mask=False,
        weights: bool = False,
    ):
        self.weights = weights
        self.root_dir = root_dir  # the directory with all the training samples
        self.num_samples = len(os.listdir(self.root_dir)) // 3  # list the samples
        self.return_mask = return_mask
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = v2.Compose(
            [
                v2.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )
        self.to_img = v2.Lambda(lambda x: torch.from_numpy(x))

        self.loaded_imgs = [None] * self.num_samples
        self.loaded_masks = [None] * self.num_samples
        for sample_ind in tqdm(range(self.num_samples)):
            img_path = os.path.join(self.root_dir, f"img_{sample_ind}.tif")
            image = self.to_img(tifffile.imread(img_path))
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            mask_path = os.path.join(self.root_dir, f"img_{sample_ind}_cyto_masks.tif")
            mask = self.to_img(tifffile.imread(mask_path))
            self.loaded_masks[sample_ind] = mask

    # get the total number of samples
    def __len__(self):
        return self.num_samples

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # We'll be using the Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        aff_mask = self.create_aff_target(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)

        if self.weights:
            weight = torch.zeros_like(aff_mask)
            for channel in range(weight.shape[0]):
                weight[channel][aff_mask[channel] == 0] = np.clip(
                    weight[channel].numel()
                    / 2
                    / (weight[channel].numel() - weight[channel].sum()),
                    0.1,
                    10.0,
                )
                weight[channel][aff_mask[channel] == 1] = np.clip(
                    weight[channel].numel() / 2 / weight[channel].sum(), 0.1, 10.0
                )

            if self.return_mask is True:
                return image, mask, aff_mask, weight
            else:
                return image, aff_mask, weight
        else:
            if self.return_mask is True:
                return image, mask, aff_mask
            else:
                return image, aff_mask

    def create_aff_target(self, mask):
        aff_target_array = compute_affinities(
            np.asarray(mask), [[0, 1], [1, 0], [0, 5], [5, 0]]
        )
        aff_target = torch.from_numpy(aff_target_array)
        return aff_target.float()


# %% [markdown]
# Next we initialize the datasets and data loaders.
# %%
# Initialize the datasets

train_data = AffinityDataset("tissuenet_data/train", v2.RandomCrop(256), weights=True)
train_loader = DataLoader(
    train_data, batch_size=5, shuffle=True, num_workers=NUM_THREADS
)
idx = np.random.randint(len(train_data))  # take a random sample
img, affinity, weight = train_data[idx]  # get the image and the nuclei masks
plot_two(img, affinity, label="AFFINITY")


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 4.1</b>: Train a model with affinities as targets.
# </div>
# %% [markdown]
# Repurpose the training loop which you used for the SDTs. <br>
# Think carefully about your final activation and number of out channels. <br>
# (The best for SDT is not necessarily the best for affinities.)

# %% tags=["task"]

unet = ...
learning_rate = ...
loss = ...
optimizer = ...

# train

# %% tags=["solution"]

unet = UNet(
    depth=4,
    in_channels=2,
    out_channels=4,
    final_activation=torch.nn.Sigmoid(),
    num_fmaps=16,
    fmap_inc_factor=3,
    downsample_factor=2,
    padding="same",
)

learning_rate = 1e-4

# choose a loss function
loss = torch.nn.BCELoss(reduce=False)

optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

val_data = AffinityDataset("tissuenet_data/test", v2.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)

# %%
for epoch in range(NUM_EPOCHS):
    train(
        unet,
        train_loader,
        optimizer,
        loss,
        epoch,
        log_interval=2,
        device=device,
    )

# %% [markdown]
# Let's next look at a prediction on a random image.

# %%
val_data = AffinityDataset("tissuenet_data/test", v2.RandomCrop(256), return_mask=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)

unet.eval()
idx = np.random.randint(len(val_data))  # take a random sample
image, gt, affs = val_data[idx]  # get the image and the nuclei masks
image = image.to(device)
pred = torch.squeeze(unet(torch.unsqueeze(image, dim=0)))
image = image.cpu()
affs = affs.cpu().numpy()
pred = pred.cpu().detach().numpy()
gt_labels = gt.cpu().numpy()

bias_short = -0.9
bias_long = -0.95

pred_labels = mws.agglom(
    np.array(
        [
            pred[0] + bias_short,
            pred[1] + bias_short,
            pred[2] + bias_long,
            pred[3] + bias_long,
        ]
    ).astype(np.float64),
    [[0, 1], [1, 0], [0, 5], [5, 0]],
)

plot_four(image, affs, pred, pred_labels, label="Affinity", cmap=label_cmap)
print(evaluate(gt_labels, pred_labels))
pred_labels = remove_small_objects(pred_labels.astype(np.int64), min_size=64, connectivity=1)
print(evaluate(gt_labels, pred_labels))
plot_four(image, affs, pred, pred_labels, label="Affinity", cmap=label_cmap)
# %% [markdown]
# Let's also evaluate the model performance.

# %%

val_dataset = AffinityDataset("tissuenet_data/test", return_mask=True)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=NUM_THREADS
)
unet.eval()

(
    precision_list,
    recall_list,
    accuracy_list,
) = (
    [],
    [],
    [],
)
for idx, (image, mask, _) in enumerate(tqdm(val_dataloader)):
    image = image.to(device)

    pred = unet(image)

    image = np.squeeze(image.cpu())

    gt_labels = np.squeeze(mask.cpu().numpy())

    pred = np.squeeze(pred.cpu().detach().numpy())

    # # feel free to try different thresholds
    # thresh = threshold_otsu(pred)

    # # get boundary mask
    # inner_mask = 0.5 * (pred[0] + pred[1]) > thresh

    # boundary_distances = distance_transform_edt(inner_mask)

    # pred_labels = watershed_from_boundary_distance(
    #     boundary_distances, inner_mask, id_offset=0, min_seed_distance=20
    # )
    pred_labels = mws.agglom(
        np.array(
            [
                pred[0] - bias_short,
                pred[1] - bias_short,
                pred[2] - bias_long,
                pred[3] - bias_long,
            ]
        ).astype(np.float64),
        [[0, 1], [1, 0], [0, 5], [5, 0]],
    )

    precision, recall, accuracy = evaluate(gt_labels, pred_labels)
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)

print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean Accuracy is {np.mean(accuracy_list):.3f}")

# %% [markdown]
# <hr style="height:2px;">
#
# ## Bonus: Further reading on Affinities
# [Here](https://localshapedescriptors.github.io/) is a blog post describing the Local Shape Descriptor method of instance segmentation.
#
# %% [markdown]
# <hr style="height:2px;">
#
# ## Bonus: Pre-Trained Models
# Cellpose has an excellent pre-trained model for instance segmentation of cells and nuclei.
# <br> take a look at the full built-in models and try to apply one to the dataset used in this exercise.
# <br> -[cellpose github](https://github.com/MouseLand/cellpose)
# <br> -[cellpose documentation](https://cellpose.readthedocs.io/en/latest/)
#
#
# %%
# Install cellpose.
# !pip install cellpose

# %% tags=["solution"]
from cellpose import models

model = models.Cellpose(model_type="nuclei")
channels = [[0, 0]]

precision_list, recall_list, accuracy_list = [], [], []
for idx, (image, mask, _) in enumerate(tqdm(val_loader)):
    gt_labels = np.squeeze(mask.cpu().numpy())
    image = np.squeeze(image.cpu().numpy())
    pred_labels, _, _, _ = model.eval([image], diameter=None, channels=channels)

    precision, recall, accuracy = evaluate(gt_labels, pred_labels[0])
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)

print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean Accuracy is {np.mean(accuracy_list):.3f}")

# %%
