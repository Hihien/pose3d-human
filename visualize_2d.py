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


def plot_pose2d(imgs, pose2d, frame_ids, fps=10):
    import matplotlib.pyplot as plt
    # from matplotlib.animation import FuncAnimation
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]
    colormap_index = np.linspace(0, 1, len(joint_pairs))

    i = 0
    for frame_id, img in enumerate(imgs):
        axes[0].clear()
        axes[1].clear()

        axes[1].invert_yaxis()
        axes[1].grid()
        axes[0].imshow(img, aspect='auto')
        if np.any(frame_ids == frame_id):
            pts = pose2d[i]
            for cm_ind, jp in zip(colormap_index, joint_pairs):
                if jp[0] > 10 or jp[1] > 10:
                    continue
                axes[0].plot(pts[jp, 0], pts[jp, 1],
                             linewidth=3.0, alpha=0.7, color=plt.cm.cool(cm_ind))
                axes[0].scatter(pts[jp, 0], pts[jp, 1], s=10)

                axes[1].plot(pts[jp, 0], pts[jp, 1],
                             linewidth=3.0, alpha=0.7, color='k')
                axes[1].scatter(pts[jp, 0], pts[jp, 1], s=30)
        i += 1
        plt.pause(1. / fps)

    # anim = matplotlib.animation.FuncAnimation(fig,
    #                                           update,
    #                                           frames=len(imgs),
    #                                           interval=1000.0 / 5, repeat=False)
    # khó
    plt.show()


def plot_pose3d(imgs, pose3d, frame_ids, fps=10):
    import matplotlib.pyplot as plt
    # from matplotlib.animation import FuncAnimation
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]
    colormap_index = np.linspace(0, 1, len(joint_pairs))

    i = 0
    for frame_id, img in enumerate(imgs):
        ax.clear()
        if np.any(frame_ids == frame_id):
            pts = pose3d[i]
            for cm_ind, jp in zip(colormap_index, joint_pairs):
                # if jp[0] > 10 or jp[1] > 10:
                #     continue
                ax.plot(pts[jp, 0], pts[jp, 1], pts[jp, 2],
                        linewidth=3.0, alpha=0.7, color=plt.cm.cool(cm_ind))
                ax.scatter(pts[jp, 0], pts[jp, 1], pts[jp, 2], s=10)
        i += 1
        plt.pause(1. / fps)

    plt.show()


def main(args):

    input_dir = 'outputs/pose'
    image_dir = 'D:/code/preprocess/data/CoDung/vis_1'
    image_files = [os.path.join(image_dir, _) for _ in sorted(os.listdir(image_dir))]
    imgs = [cv2.cvtColor(cv2.imread(_), cv2.COLOR_BGR2RGB) for _ in image_files]

    person_id = 5
    json_file = os.path.join(input_dir, f'{person_id:02d}.json')
    pose = json.load(open(json_file))
    pose2d = np.asarray(pose['2d'])
    pose3d = np.asarray(pose['3d'])
    frame_ids = np.asarray(pose['frames'])
    del pose
    print(pose2d.shape)
    print(pose3d.shape)
    print(frame_ids)
    plot_pose2d(imgs, pose2d, frame_ids)
    # plot_pose3d(imgs, pose3d, frame_ids)


if __name__ == '__main__':
    args = parse_args()
    main(args)
