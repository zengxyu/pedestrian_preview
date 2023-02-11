def compute_conv_out_width(i, k, s, p, iter=1):
    """
    计算卷积输出层的宽度
    compute the width of convolution output
    :param i: 输入尺寸 input size
    :param k: 卷积核大小 kernel size
    :param s: 步幅 stride
    :param p: 边界扩充 padding
    :return: 输出的feature map的宽 the width of convolution output
    """
    for _ in range(iter):
        i = (i - k + 2 * p) / s + 1
    o = int(i)
    return int(o)


def compute_de_conv_out_width(i, k, s, p):
    """
    计算反卷积输出层的宽度
    compute the width of transpose convolution output
    :param i: 输入尺寸 input size
    :param k: 卷积核大小 kernel size
    :param s: 步幅 stride
    :param p: 边界扩充 padding
    :return: 输出的feature map的宽 the width of convolution output
    """
    out = (i - 1) * s + k - 2 * p
    return int(out)


def compute_conv_out_node_num(d, w, h):
    """
    compute the number of neuron output of convolution
    :param d: depth channel number
    :param w: width
    :param h: height
    :return:
    """
    return int(d * w * h)


if __name__ == '__main__':
    a = compute_conv_out_width(i=20, k=3, s=2, p=1)
    a = compute_conv_out_width(i=a, k=3, s=2, p=1)
    a = compute_conv_out_width(i=a, k=3, s=2, p=1)

    print(a)
