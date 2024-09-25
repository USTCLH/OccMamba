import os
import yaml
import numpy as np
from tqdm import tqdm

def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def pack(array):
    """ convert a boolean array into a bitwise array. """
    array = array.reshape((-1))

    #compressing bit flags.
    # yapf: disable
    compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
    # yapf: enable

    return np.array(compressed, dtype=np.uint8)

if __name__ == "__main__":
    # input_path = 'data/kitti/voxel_labels'
    input_path = 'output/for_submit/voxel_labels'
    # input_path = 'backups/test'
    output_path = 'output/submit/sequences'
    test_submit = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    # test_submit = ['08']

    DATA = yaml.safe_load(open('tools/kitti_submit/semantic-kitti.yaml', 'r'))
    class_inv_remap = DATA["learning_map_inv"]
    maxkey = max(class_inv_remap.keys())
    inv_remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    inv_remap_lut[list(class_inv_remap.keys())] = list(class_inv_remap.values())

    for seq in test_submit:
        seq_input_path = os.path.join(input_path, seq)
        seq_output_path = os.path.join(output_path, seq, 'predictions')

        os.makedirs(seq_output_path, exist_ok=True)
        file_list = sorted(os.listdir(seq_input_path))

        for file in tqdm(file_list):
            pred = np.load(os.path.join(seq_input_path, file)).astype(np.uint16)
            pred = pred.reshape(-1)
            pred = inv_remap_lut[pred].astype(np.uint16)
            # pred[pred == 255] = 0
            # asd
            pred.tofile(os.path.join(seq_output_path, file.replace("_1_1.npy", ".label")))