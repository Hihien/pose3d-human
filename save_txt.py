import json

from common.arguments import *
from common.camera import *
from common.generators import *
from common.model import *
from common.utils import *
from common.visualization import *


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]


def main(args):
    pose_kpt_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                      [14, 128, 250], [80, 127, 255], [80, 127, 255],
                      [80, 127, 255], [80, 127, 255], [71, 99, 255],
                      [71, 99, 255], [71, 99, 255], [71, 99, 255],
                      [0, 36, 255], [0, 36, 255], [0, 36, 255],
                      [0, 36, 255], [0, 0, 230], [0, 0, 230], [0, 0, 230],
                      [0, 0, 230], [0, 0, 139], [237, 149, 100],
                      [237, 149, 100], [237, 149, 100], [237, 149, 100],
                      [230, 128, 77], [230, 128, 77], [230, 128, 77],
                      [230, 128, 77], [255, 144, 30], [255, 144, 30],
                      [255, 144, 30], [255, 144, 30], [153, 51, 0],
                      [153, 51, 0], [153, 51, 0], [153, 51, 0],
                      [255, 51, 13], [255, 51, 13], [255, 51, 13],
                      [255, 51, 13], [103, 37, 8]]
    json_file = './id_track_1.json'
    image_dir = 'D:/code/preprocess/data/CoDung/16_images'
    output_dir = 'outputs/pose'
    os.makedirs(output_dir, exist_ok=True)

    metadata = {'layout_name': 'coco', 'num_joints': 17,
                'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}
    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout,
                              channels=args.channels,
                              dense=args.dense)
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)  # 把loc映射到storage
    model_pos.load_state_dict(checkpoint['model_pos'])

    annotations = json.load(open(json_file))
    person_ids = np.unique([_['idx'] for _ in annotations])
    print(person_ids)

    for person in person_ids:
        all_keypoints = []
        frame_ids = []
        for frame_id, img_file in enumerate(sorted(os.listdir(image_dir))):
            skeletons = list(filter(lambda _: _['image_id'] == img_file, annotations))
            for skeleton in skeletons:
                person_id = skeleton['idx']
                if person_id != person:
                    continue

                keypoints = np.asarray(skeleton['keypoints']).reshape(-1, 3)
                keypoints = keypoints[:, :-1].round().astype(int)
                all_keypoints.append(keypoints)
                frame_ids.append(frame_id)

        all_keypoints = np.asarray(all_keypoints)  # (N, 17, 2)
        all_keypoints = normalize_screen_coordinates(all_keypoints[..., :2], w=1000, h=1002)
        frame_ids = np.asarray(frame_ids)
        if not len(all_keypoints):
            continue

        #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
        receptive_field = model_pos.receptive_field()
        pad = (receptive_field - 1) // 2  # Padding on each side
        causal_shift = 0

        input_keypoints = all_keypoints.copy()
        gen = UnchunkedGenerator(None, None, [input_keypoints],
                                 pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                 joints_right=joints_right)
        prediction = evaluate(gen, model_pos, return_predictions=True)

        rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
        prediction = camera_to_world(prediction, R=rot, t=0)

        prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

        print(person, input_keypoints.shape, prediction.shape, frame_ids)
        save_file = os.path.join(output_dir, f'{person:02d}.json')
        json.dump({'2d': input_keypoints.tolist(),
                   '3d': prediction.tolist(),
                   'frames': frame_ids.tolist()},
                  open(save_file, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)


if __name__ == '__main__':
    args = parse_args()
    main(args)
