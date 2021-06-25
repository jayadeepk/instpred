# Network layers

## F2Fi_l Model Size

    Autoregressive(
        (single_frame_model): F2Fi_multiscale(
            (models): ModuleList(
                (0): F2Fi_l(
                    (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
                    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
                    (conv3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
                    (conv4): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
                    (conv5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
                    (conv6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
                    (conv7): Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
                )
            )
        )
    )
    +------------------------------------------+------------+
    |                 Modules                  | Parameters |
    +------------------------------------------+------------+
    | single_frame_model.models.0.conv1.weight |   524288   |
    |  single_frame_model.models.0.conv1.bias  |    512     |
    | single_frame_model.models.0.conv2.weight |  2359296   |
    |  single_frame_model.models.0.conv2.bias  |    512     |
    | single_frame_model.models.0.conv3.weight |  2359296   |
    |  single_frame_model.models.0.conv3.bias  |    512     |
    | single_frame_model.models.0.conv4.weight |  1179648   |
    |  single_frame_model.models.0.conv4.bias  |    256     |
    | single_frame_model.models.0.conv5.weight |   589824   |
    |  single_frame_model.models.0.conv5.bias  |    256     |
    | single_frame_model.models.0.conv6.weight |   589824   |
    |  single_frame_model.models.0.conv6.bias  |    256     |
    | single_frame_model.models.0.conv7.weight |  3211264   |
    |  single_frame_model.models.0.conv7.bias  |    256     |
    +------------------------------------------+------------+
    Total Trainable Params: 10816000

