import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Linear, InstanceNorm, Sequential, SpectralNorm


#################################################################
#                        torch函数扩展实现                      #
#################################################################
class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, pad):
        super(ReflectionPad2d, self).__init__()
        self.pad = pad
    
    def forward(self, x):
        output = fluid.layers.pad2d(input=x, paddings=[self.pad, self.pad, self.pad, self.pad], mode='reflect')
        return output


class ReLU(fluid.dygraph.Layer):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        if self.inplace:
            x.set_value(fluid.layers.relu(x))
            return x
        else:
            y = fluid.layers.relu(x)
            return y


class Tanh(fluid.dygraph.Layer):
    def __init__(self):
        super(Tanh, self).__init__()
    
    def forward(self, inputs):
        out = fluid.layers.tanh(x=inputs)
        return out


class LeakyReLU(fluid.dygraph.Layer):
    def __init__(self, alpha=0.02, inplace=False):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace
    
    def forward(self, x):
        if self.inplace:
            x.set_value(fluid.layers.leaky_relu(x=x, alpha=self.alpha))
            return x
        else:
            return fluid.layers.leaky_relu(x=x, alpha=self.alpha)


def var(input, axis=None, keepdim=False, unbiased=True, out=None, name=None):
    dtype = fluid.data_feeder.convert_dtype(input.dtype)
    if dtype not in ["float32", "float64"]:
        raise ValueError("Layer tensor.var() only supports floating-point "
                         "dtypes, but received {}.".format(dtype))
    rank = len(input.shape)
    axes = axis if axis is not None and axis != [] else range(rank)
    axes = [e if e >= 0 else e + rank for e in axes]
    inp_shape = input.shape if fluid.framework.in_dygraph_mode() else fluid.layers.shape(input)
    mean = fluid.layers.reduce_mean(input, dim=axis, keep_dim=True, name=name)
    tmp = fluid.layers.reduce_mean(
        (input - mean)**2, dim=axis, keep_dim=keepdim, name=name)

    if unbiased:
        n = 1
        for i in axes:
            n *= inp_shape[i]
        if not fluid.framework.in_dygraph_mode():
            n = fluid.layers.cast(n, dtype)
            zero_const = fluid.layers.fill_constant(shape=[1], dtype=dtype, value=0.0)
            factor = paddle.fluid.layers.where(n > 1.0, n / (n - 1.0), zero_const)
        else:
            factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    if out:
        fluid.layers.assign(input=tmp, output=out)
        return out
    else:
        return tmp


class Spectralnorm(fluid.dygraph.Layer):
    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out


#################################################################
#                           功能模块定义                        #
#################################################################
class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(pad=1),
                       Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim),
                       ReLU(inplace=True)]
        
        conv_block += [ReflectionPad2d(pad=1),
                       Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim)]
        
        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.Constant(0.9))

    def forward(self, input, gamma, beta):
        in_mean = fluid.layers.reduce_mean(input=input, dim=[2, 3], keep_dim=True)
        in_var = var(input=input, axis=[1, 2, 3], keepdim=True)
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean = fluid.layers.reduce_mean(input=input, dim=[1, 2, 3], keep_dim=True)
        ln_var = var(input=input, axis=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        
        out = fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1]) * out_in + (1-fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1])) * out_ln
        gamma = fluid.layers.unsqueeze(input=gamma, axes=[2, 3])
        beta = fluid.layers.unsqueeze(input=beta, axes=[2, 3])
        out = out * gamma + beta

        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2d(pad=1)
        self.conv1 = Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU(True)

        self.pad2 = ReflectionPad2d(pad=1)
        self.conv2 = Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)
    
    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


# 定义上采样模块
class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        out = fluid.layers.resize_nearest(input=inputs, scale=self.scale, actual_shape=out_shape)
        return out


class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.Constant(0.0))
        self.gamma = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.Constant(1.0))
        self.beta = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.Constant(0.0))
    
    def forward(self, input):
        in_mean = fluid.layers.reduce_mean(input=input, dim=[2, 3], keep_dim=True)
        in_var = var(input=input, axis=[2, 3], keepdim=True)
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean = fluid.layers.reduce_mean(input=input, dim=[1, 2, 3], keep_dim=True)
        ln_var = var(input=input, axis=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        
        out = fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1]) * out_in + (1-fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1])) * out_ln
        out = out * fluid.layers.expand(x=self.gamma, expand_times=[input.shape[0], 1, 1, 1]) + fluid.layers.expand(x=self.beta, expand_times=[input.shape[0], 1, 1, 1])

        return out


class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = fluid.layers.clip(x=w, min=self.clip_min, max=self.clip_max)
            module.rho.data = w


#################################################################
#                           模型 构建                           #
#################################################################
class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [ReflectionPad2d(pad=3),
                      Conv2D(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(num_channels=ngf),
                      ReLU(inplace=True)]
        
        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [ReflectionPad2d(pad=1),
                          Conv2D(num_channels=ngf*mult, num_filters=ngf*mult*2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(num_channels=ngf*mult*2),
                          ReLU(inplace=True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]
        
        # Class Activation Map
        self.gap_fc = Linear(input_dim=ngf*mult, output_dim=1, bias_attr=False)
        self.gmp_fc = Linear(input_dim=ngf*mult, output_dim=1, bias_attr=False)
        self.conv1x1 = Conv2D(num_channels=ngf*mult*2, num_filters=ngf*mult, filter_size=1, stride=1, bias_attr=True)
        self.relu = ReLU(inplace=True)

        # Gamma, Beta block
        if self.light:
            FC = [Linear(input_dim=ngf*mult, output_dim=ngf*mult, bias_attr=False),
                  ReLU(inplace=True),
                  Linear(input_dim=ngf*mult, output_dim=ngf*mult, bias_attr=False),
                  ReLU(True)]
        else:
            FC = [Linear(input_dim=img_size//mult*img_size//mult*ngf*mult, output_dim=ngf*mult, bias_attr=False),
                  ReLU(inplace=True),
                  Linear(input_dim=ngf*mult, output_dim=ngf*mult, bias_attr=False),
                  ReLU(True)]
        
        self.gamma = Linear(input_dim=ngf*mult, output_dim=ngf*mult, bias_attr=False)
        self.beta = Linear(input_dim=ngf*mult, output_dim=ngf*mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [Upsample(scale=2),
                         ReflectionPad2d(pad=1),
                         Conv2D(num_channels=ngf * mult, num_filters=int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                         ILN(int(ngf * mult / 2)),
                         ReLU(True)]
        
        UpBlock2 += [ReflectionPad2d(pad=3),
                     Conv2D(num_channels=ngf, num_filters=output_nc, filter_size=7, stride=1, padding=0, bias_attr=False),
                     Tanh()]
        
        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)


    def forward(self, input):
        x = self.DownBlock(input)

        gap = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
        gap = fluid.layers.reshape(x=gap, shape=[x.shape[0], -1])
        gap_logit = self.gap_fc(gap)
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.reshape(x=gap_weight, shape=[gap_weight.shape[1], gap_weight.shape[0]])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[2, 3])
        gap = x * gap_weight

        gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='max')
        gmp = fluid.layers.reshape(x=gmp, shape=[x.shape[0], -1])
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.reshape(x=gmp_weight, shape=[gmp_weight.shape[1], gmp_weight.shape[0]])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[2, 3])
        gmp = x * gmp_weight

        cam_logit = fluid.layers.concat(input=[gap_logit, gmp_logit], axis=1)
        x = fluid.layers.concat(input=[gap, gmp], axis=1)
        x = self.relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(input=x, dim=1, keep_dim=True)

        if self.light:
                x_ = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
                x_ = fluid.layers.reshape(x=x_, shape=[x_.shape[0], -1])
                x_ = self.FC(x_)
        else:
            x = fluid.layers.reshape(x=x, shape=[x.shape[0], -1])
            x_ = self.FC(x)
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2d(pad=1),
                 Spectralnorm(
                 Conv2D(num_channels=input_nc, num_filters=ndf, filter_size=4, stride=2, padding=0, bias_attr=True)),
                 LeakyReLU(alpha=0.2, inplace=True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2d(pad=1),
                      Spectralnorm(
                      Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=2, padding=0, bias_attr=True)),
                      LeakyReLU(alpha=0.2, inplace=True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2d(pad=1),
                  Spectralnorm(
                  Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=1, padding=0, bias_attr=True)),
                  LeakyReLU(alpha=0.2, inplace=True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(Linear(input_dim=ndf * mult, output_dim=1, bias_attr=False))
        self.gmp_fc = Spectralnorm(Linear(input_dim=ndf * mult, output_dim=1, bias_attr=False))
        self.conv1x1 = Conv2D(num_channels=ndf * mult * 2, num_filters=ndf * mult, filter_size=1, stride=1, bias_attr=True)
        self.leaky_relu = LeakyReLU(alpha=0.2, inplace=True)

        self.pad = ReflectionPad2d(pad=1)
        self.conv = Spectralnorm(
            Conv2D(num_channels=ndf * mult, num_filters=1, filter_size=4, stride=1, padding=0, bias_attr=False))

        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
        gap = fluid.layers.reshape(x=gap, shape=[x.shape[0], -1])
        gap_logit = self.gap_fc(gap)
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.reshape(x=gap_weight, shape=[gap_weight.shape[1], gap_weight.shape[0]])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[2, 3])
        gap = x * gap_weight

        gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='max')
        gmp = fluid.layers.reshape(x=gmp, shape=[x.shape[0], -1])
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.reshape(x=gmp_weight, shape=[gmp_weight.shape[1], gmp_weight.shape[0]])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[2, 3])
        gmp = x * gmp_weight

        cam_logit = fluid.layers.concat(input=[gap_logit, gmp_logit], axis=1)
        x = fluid.layers.concat(input=[gap, gmp], axis=1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(input=x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


