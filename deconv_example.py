import torch
import torch.nn as t


def main():
    deconv = t.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=True, bias=False)
    conv = t.Conv2d(4, 1, kernel_size=3, stride=1, padding=True, bias=False)
    a = torch.randn((3, 1, 256, 256))
    a = a + a
    print('a.shape:', a.shape)
    deconv_a = deconv(a)
    print('deconv_a.shape:', deconv_a.shape)
    reshaped_a = deconv_a.reshape(3, -1, 128, 512)
    deconv_b = conv(reshaped_a)
    print('deconv_b.shape:', deconv_b.shape)


def main_conv2():

    deconv = t.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=True, bias=False)
    conv = t.Conv2d(4, 1, kernel_size=3, stride=1, padding=True, bias=False)
    conv1d = t.Conv1d(384, 129, kernel_size=3, stride=1, padding=True, bias=False)
    conv1d_downsample = t.Conv1d(258, 129, kernel_size=3, stride=1, padding=True, bias=False)
    a = torch.randn((3, 1, 256, 256))
    c = torch.randn(16, 384, 96)

    c_out = conv1d(c)
    c_out_reshape = c_out.reshape(16, 1, 129, 96)
    c_out_reshape_out = deconv(c_out_reshape)
    c_out_reshape_out_reshape = c_out_reshape_out.reshape(16, 258, 192)
    c_out_reshape_out_reshape_down = conv1d_downsample(c_out_reshape_out_reshape)

    print('c_cout.shape:', c_out.shape)
    print('c.shape:', c.shape)
    print('c_out_reshape_out.shape:', c_out_reshape_out.shape)
    print('c_out_reshape_out_reshape.shape:', c_out_reshape_out_reshape.shape)
    print('c_out_reshape_out_reshape_down.shape:', c_out_reshape_out_reshape_down.shape)


if __name__ == '__main__':
    # main()
    main1()





