
# 防止出现griding问题，计算膨胀率
def dilate_rate(kernel=None):
    k = kernel
    result = []
    for i in range(1, 10):
        for j in range(1, 10):
            for v in range(1, 10):
                temp = max(v - 2 * j, 2 * j - v, j)
                if temp <= k and (i <= j) and (j <= v) and not (i / j == j / v):
                    rate = str(i) + str(j) + str(v)
                    result.append(rate)
    print('========dilate_rate========')
    for index, val in enumerate(result, 1):
        print(val, end='--')
        if index % 5 == 0:
            print()
# dilate_rate(3)

# 计算特征图的感受野
def receptive_field():
    net_kernel_stride = [
        # 1 [kernel, stride], kernel is dilated kernel
        # D : dilated ratio
        # Kd = (k-1)*D + 1
        # [3, 1], [3, 1], [3, 1],
        # 2
        # [3, 1], [5, 1], [7, 1],
        # # 3
        [3, 1], [7, 1], [11, 1]
    ]
    ks_pairs = net_kernel_stride
    # K: composed kernel, also the receptive field
    # S: composed stride
    K, S = 1, 1
    layers = range(len(ks_pairs))
    for layer, ks_pair in zip(layers, ks_pairs):
        k, s = ks_pair
        K = (k - 1) * S + K
        S = S * s
        print('layer {:<15}: {} [{:3},{:2}]'.format(layer, ks_pair, K, S))
receptive_field()