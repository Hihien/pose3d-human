import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import json
import os
from common.camera import *
from common.model import *
from common.utils import *
from common.visualization import *
from common.arguments import *
from common.generators import *

class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]

def main(args):
    radius =4
    json_file = './id_track_1.json'
    image_dir = 'D:/code/preprocess/data/CoDung/vis_1'

    metadata = {'layout_name': 'coco', 'num_joints': 17,
                'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}
    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    annotations = json.load(open(json_file))

    all_keypoints = []
    time0 = ckpt_time()

    for img_file in tqdm(sorted(os.listdir(image_dir))):
        skeletons = list(filter(lambda _: _['image_id'] == img_file, annotations))
        image = cv2.imread(os.path.join(image_dir, img_file))
        img_copy = image.copy()
        for person_id, skeleton in enumerate(skeletons):
            person_id = str(skeleton['idx'])
            if (person_id == '8'):
                keypoints = np.asarray(skeleton['keypoints']).reshape(-1, 3)
                keypoints = keypoints[:, :-1].round().astype(int)
                all_keypoints.append(keypoints)
                all = np.asarray(all_keypoints)   # (N, 17, 2)

    all = normalize_screen_coordinates(all[..., :2], w=1000, h=1002)
    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels,
                          dense=args.dense)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    ckpt, time1 = ckpt_time(time0)
    print('--------------load data spend {:.2f} second'.format(ckpt))

    # load trained model
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)  # 把loc映射到storage
    model_pos.load_state_dict(checkpoint['model_pos'])

    ckpt, time2 = ckpt_time(time1)
    print('-------------- load 3D model spends {:.2f} seconds'.format(ckpt))

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    input_keypoints = all.copy()
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                             joints_right=joints_right)
    prediction = evaluate(gen, model_pos, return_predictions=True)
    np.save('outputs/skeletons.npy', prediction, allow_pickle=True)

    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    prediction = camera_to_world(prediction, R=rot, t=0)

    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    anim_output = {'Reconstruction': prediction}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)
    print(input_keypoints[0])
    print(prediction[0])

    ckpt, time3 = ckpt_time(time2)
    print('-------------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))

    if not args.viz_output:
        args.viz_output = 'outputs/results.gif'
    render_animation(input_keypoints, anim_output,
                    Skeleton(), 5, args.viz_bitrate, np.array(70., dtype=np.float32), args.viz_output,
                    limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                    input_video_path=args.viz_video, viewport=(1000, 1002),
                    input_video_skip=args.viz_skip)

    ckpt, time4 = ckpt_time(time3)
    print('total spend {:2f} second'.format(ckpt))


if __name__ == '__main__':
    args = parse_args()
    main(args)
