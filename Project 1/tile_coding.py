import numpy as np


def create_single_tiling(
        feat_range: list,
        bins: int,
        offset: float) -> np.ndarray:
    """Create a single tiling that corresponds to the tile of a single feature

    Args:
        feat_range (np.ndarray): Array that contains ranges of values for the feature (low - high)
        bins (int): The number of different bins/buckets to put the feature into
        offset (float): Offset of the bin-values, i.e. we can "nudge" the bin in a certain direction

    Returns:
        np.ndarray: Array that contains the tiles for the feature, where the tile is evenly divided into bins
    """
    # a = np.linspace(feat_range[0], feat_range[1], bins - 1)
    # b = a[1:-1]
    # c = b + offset
    # print(a, b, c)
    return np.linspace(feat_range[0], feat_range[1], bins-1) + offset


def create_tilings(feature_ranges: list, number_tilings: int, bins: list, offsets: list) -> np.ndarray:
    """Create an array of multiple tilings for every feature, with every tile having individual bins and offsets.
    So you can send in say 2 features, with ranges = [[0, 1], [5, 10]], meaning feat1 has range 0-1, and feat2 5-10.
    And say you have number_tilings = 2. Then we say that we want to have 2 tiles for 2 features. We can then send
    in the following:
        bins=[[10, 5], [4, 8]]
        offsets=[[0.1, 0.3], [0.5, 0.1]]
    What does this mean? Well the first tile will get the following configuration:
        feat1 is divided into 10 bins, feat2 is divided into 5 bins
        feat1's bins are offset by 0.1, feat2's bins are offset by 0.3
    So every array contains the bins for the respective variables and the offsets of those bins for every feature
    This will be used to represent the states for the RL agent

    Args:
        feature_ranges (list): A 2-dimensional array containing the ranges of every feature
        number_tilings (int): An integer describing the number of tiles wanted
        bins (list): A 2-dimensional array with n-dimensional internal arrays, where n is the number of features
        offsets (list): A 2-dimensional array with n-dimensional internal arrays, where n is the number of features

    Returns:
        np.ndarray: A tile encoding of the features, given their ranges, bins, and offsets
    """
    tilings = []
    # for each tiling
    for tile_i in range(number_tilings):
        tiling_bin = bins[tile_i]
        tiling_offset = offsets[tile_i]

        tiling = []
        # for each feature dimension
        for feat_i in range(len(feature_ranges)):
            feat_range = feature_ranges[feat_i]
            # tiling for 1 feature
            feat_tiling = create_single_tiling(
                feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
            tiling.append(feat_tiling)
        tilings.append(tiling)
    for coding in tilings:
        if len(coding) > 1:
            result = coding[1]
            for i in range(1, len(coding)):
                result = np.concatenate((result, coding[i]), axis=None)
    return np.array(tilings, dtype='object')


def get_state_encoding(feature_values: list, tilings: list) -> np.ndarray:
    """The final function in the tile-coding hierarchy; this function takes in a set of features and its corresponding tiles,
    and outputs the tile-encoding for that particular set of features.
    In context of the RL agent, the tiles are the states, and the state-space consists of all possible configurations of the tiles.

    Args:
        feature_values (list): An array containing exact values for all the n features
        tilings (list): An array representing the tilings for the features, with associated bins and offsets for each feature

    Returns:
        np.ndarray: An array containing the tile-encoding of the particular features
    """
    num_dims = len(feature_values)
    feat_codings = []
    for tiling in tilings:
        feat_coding = []
        for i in range(num_dims):
            feat_i = feature_values[i]
            tiling_i = tiling[i]  # tiling on that dimension
            coding_i = np.digitize(feat_i, tiling_i)
            feat_coding.append(coding_i)
        feat_codings.append(feat_coding)
    return np.array(feat_codings)
    

def test_pole_tilings() -> None:
    # Features in following order: cart position, cart velocity, angle position, angle velocity, timestep
    cart_pos_range = [-2.4, 2.4]
    cart_speed_range = [-2, 2]
    angle_pos_range = [-0.21, 0.21]
    angle_vel_range = [-2, 2]
    timestep_range = [0, 300]
    feature_ranges = [cart_pos_range, cart_speed_range, angle_pos_range, angle_vel_range, timestep_range]
    n_tilings = 1
    # Configure the number of bins and offsets for each feature
    cart_pos_bins = 4
    cart_pos_offset = 0
    cart_speed_bins = 4
    cart_speed_offset = 0
    angle_pos_bins = 8
    angle_pos_offset = 0
    angle_vel_bins = 8
    angle_vel_offset = 0
    # Timestep gets 2 bins, because its only purpose is to determine whether or not we're in a winning state
    timestep_bins = 2
    # Offset = 150 => timestep <= 299 => we're not winning, timestep >= 300 => we are winning
    # This because if tau <= 299, the tile encoding will result in 0 with this offset
    # (Reason for 150: linspace takes range/2 = 150, so offset 150 => bin-value becomes 300)
    timestep_offset = 300
    feat_bins = [[cart_pos_bins, cart_speed_bins, angle_pos_bins, angle_vel_bins, timestep_bins]]
    feat_offsets = [[cart_pos_offset, cart_speed_offset, angle_pos_offset, angle_vel_offset, timestep_offset]]
    test_tilings = create_tilings(feature_ranges, n_tilings, feat_bins, feat_offsets)
    print(test_tilings)    
    test_encoding = [1.2, -100, -0.3, 2, 300]
    test_result = get_state_encoding(test_encoding, test_tilings)
    print(test_result)

def test_other_thing() -> None:
    feature_ranges = [[-2.4, 2.4], [-1, 1], [-0.21, 0.21], [-0.1, 0.1], [0, 300]]
    number_tilings = 2
    bins = [[4, 4, 8, 8, 2], [5, 5, 9, 9, 2]]
    offsets = [[0, 0, 0, 0, 300], [1, 1, 1, 1, 300]]

    tilings = create_tilings(feature_ranges, number_tilings, bins, offsets)
    print(tilings)
    feature = [0.1, 2.5, 0.3, 0.2, 0]

    coding = get_state_encoding(feature, tilings)
    print(coding)

if __name__ == '__main__':
    test_pole_tilings()
    # test_other_thing()
