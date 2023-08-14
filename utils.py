import numpy as np
from skimage import color
from scipy.ndimage import binary_erosion, distance_transform_edt, maximum_filter
from scipy.ndimage import label as label_cc
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu
from skimage.measure import label as relabel_cc
from skimage.segmentation import watershed, relabel_sequential


def create_lut(labels: np.ndarray) -> np.ndarray:
    """Utility function to view labels as rgb lut with matplotlib.
    
    # eg plt.imshow(create_lut(labels))
    """
    max_label = np.max(labels)

    lut = np.random.randint(
            low=0,
            high=255,
            size=(int(max_label + 1), 3),
            dtype=np.uint8)

    lut = np.append(
            lut,
            np.zeros(
                (int(max_label + 1), 1),
                dtype=np.uint8) + 255,
            axis=1)

    lut[0] = 0
    colored_labels = lut[labels]

    return colored_labels


def erode(labels: np.ndarray, iterations: int, border_value: int):
    """Function to erode boundary pixels. 
    
    We fill an array with zeros, iterate over our labels, erode them
    and then write them back into our empty array
    """

    # copy labels to memory, create border array
    labels = np.copy(labels)

    # create zeros array for foreground
    foreground = np.zeros_like(labels, dtype=bool)

    # loop through unique labels
    for label in np.unique(labels):

        # skip background
        if label == 0:
            continue

        # mask to label
        label_mask = labels == label

        # erode labels
        eroded_mask = binary_erosion(
                label_mask,
                iterations=iterations,
                border_value=border_value)

        # get foreground
        foreground = np.logical_or(eroded_mask, foreground)

    # and background...
    background = np.logical_not(foreground)

    # set eroded pixels to zero
    labels[background] = 0

    return labels


def erode_border(labels, iterations, border_value):
    """Function to erode boundary pixels for mask and border."""

    # copy labels to memory, create border array
    labels = np.copy(labels)
    border = np.array(labels)

    # create zeros array for foreground
    foreground = np.zeros_like(labels, dtype=bool)

    # loop through unique labels
    for label in np.unique(labels):

        # skip background
        if label == 0:
            continue

        # mask to label
        label_mask = labels == label

        # erode labels
        eroded_mask = binary_erosion(
                label_mask,
                iterations=iterations,
                border_value=border_value)

        # get foreground
        foreground = np.logical_or(eroded_mask, foreground)

    # and background...
    background = np.logical_not(foreground)

    # set eroded pixels to zero
    labels[background] = 0

    # get eroded pixels
    border = labels - border

    return labels, border


def compute_sdt(labels: np.ndarray, constant: float = 0.5, scale: int = 5):
    """Function to compute a signed distance transform."""

    inner = distance_transform_edt(binary_erosion(labels))
    outer = distance_transform_edt(np.logical_not(labels))

    distance = (inner - outer) + constant

    distance = np.tanh(distance / scale)

    return distance


# utility function to compute edge affinities
def compute_affinities(seg: np.ndarray, nhood: list):

    nhood = np.array(nhood)

    shape = seg.shape
    nEdge = nhood.shape[0]
    dims = nhood.shape[1]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
          max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
          max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] = \
                      (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                          max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] == \
                        seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                          max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] ) \
                      * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                          max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] > 0 ) \
                      * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                          max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] > 0 )
                          

    return aff


def watershed_from_boundary_distance(
        boundary_distances: np.ndarray,
        boundary_mask: np.ndarray,
        id_offset: float = 0,
        min_seed_distance: int = 10
        ):
    """Function to compute a watershed from boundary distances."""

    # get our seeds 
    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = label_cc(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    # calculate our segmentation
    segmentation = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask)
    
    return segmentation

def get_boundary_mask(pred, prediction_type, thresh=None):
    
    if prediction_type == 'two_class' or prediction_type == 'sdt':
        # simple threshold
        boundary_mask = pred > thresh

    elif prediction_type == 'three_class':
        # Return the indices of the maximum values along channel axis, then set mask to cell interior (1)
        boundary_mask = np.argmax(pred, axis=0)
        boundary_mask = boundary_mask == 1

    elif prediction_type == 'affs':
        # take mean of combined affs then threshold
        boundary_mask = 0.5 * (pred[0] + pred[1]) > thresh
    else:
        raise Exception('Choose from one of the following prediction types: two_class, three_class, sdt, affs')
        
    return boundary_mask



def evaluate(gt_labels: np.ndarray, pred_labels: np.ndarray, th: float = 0.5):
    """Function to evaluate a segmentation."""
    
    pred_labels_rel, _, _ = relabel_sequential(pred_labels)
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    overlay = np.array([pred_labels_rel.flatten(),
                        gt_labels_rel.flatten()])

    # get overlaying cells and the size of the overlap
    overlay_labels, overlay_labels_counts = np.unique(
        overlay, return_counts=True, axis=1)
    overlay_labels = np.transpose(overlay_labels)

    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels_rel, return_counts=True)
    gt_labels_count_dict = {}
    
    for (l, c) in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c

    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels_rel,
                                              return_counts=True)

    pred_labels_count_dict = {}
    for (l, c) in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c

    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)
    
    # create iou table
    iouMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                      dtype=np.float32)

    for (u, v), c in zip(overlay_labels, overlay_labels_counts):
        iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)
        iouMat[int(v), int(u)] = iou

    # remove background
    iouMat = iouMat[1:, 1:]

    # use IoU threshold th
    if num_matches > 0 and np.max(iouMat) > th:
        costs = -(iouMat > th).astype(float) - iouMat / (2*num_matches)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        assert num_matches == len(gt_ind) == len(pred_ind)
        match_ok = iouMat[gt_ind, pred_ind] > th
        tp = np.count_nonzero(match_ok)
    else:
        tp = 0
    fp = num_pred_labels - tp
    fn = num_gt_labels - tp
    ap = tp / max(1, tp + fn + fp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    return ap, precision, recall, tp, fp, fn