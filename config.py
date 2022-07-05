import torch

class Config:
    exp = '0705_Frei'

    model_name = 'SAR_TE'  # 'SAR', 'SAR_TE'
    dataset = 'FreiHAND'
    output_root = './output'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # flags
    small_dataset = False
    load_db = False

    # network
    backbone = 'resnet34'
    num_stage = 2
    num_FMs = 8
    feature_size = 64
    heatmap_size = 32
    num_vert = 778
    num_joint = 21

    fakeheatmap_size = 64
    mode_addPos = 'Front'    # 'Front', 'Back'

    # training
    batch_size = 64
    lr = 3e-4
    weight_decay = 0.08     # defalut : 0.1
    total_epoch = 50
    input_img_shape = (256, 256)
    depth_box = 0.3
    num_worker = 16
    # -------------
    save_epoch = 1
    eval_interval = 1
    print_iter = 10
    num_epoch_to_eval = 80
    # -------------
    checkpoint = './checkpoint/SAR-R34-S2-65-67.pth'  # put the path of the trained model's weights here
    continue_train = False
    vis = True
    # -------------
    experiment_name = exp + '_{}'.format(mode_addPos) + '_{}'.format(backbone) + '_Batch{}'.format(batch_size) + \
                      '_lr{}'.format(lr) + '_Epochs{}'.format(total_epoch)

cfg = Config()
