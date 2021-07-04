import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Dense,Layer,ReLU,LeakyReLU,GlobalAvgPool2D,UpSampling2D,LayerNormalization,Activation
from tensorflow.keras import initializers

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs): #padding=(1,1) = 'same' if kernel size is 3
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0]], 'REFLECT')

class AdaIN(Layer):
    def __init__(self,in_channels,ep=1e-8):
        super(AdaIN,self).__init__()
        self.params_num=in_channels*2
        self.gamma=0
        self.beta=0
        self.ep=ep

    def call(self,x,training=False):
        '''
        x:(batch,h,w,c)
        gamma、beta:(batch,1,1,c)
        '''
        mean = tf.reduce_mean(x, keepdims=True, axis=[1, 2])  # batch,1,1,c
        std = tf.math.reduce_std(x, keepdims=True, axis=[1, 2])  # batch,1,1,c
        z = (x - mean) / (std + self.ep)
        return self.gamma * z + self.beta


class Instance_norm(Layer):
    def __init__(self,ep=1e-8):
        super(Instance_norm,self).__init__()
        self.ep=ep

    def build(self,input_shape):
        self.gamma=self.add_weight(name='gamma',
                                   shape=[1,1,int(input_shape[-1])],
                                   initializer=initializers.RandomNormal(mean=0,stddev=1),
                                   trainable=True)
        self.beta=self.add_weight(name='beta',
                                  shape=[1,1,int(input_shape[-1])],
                                  initializer=initializers.RandomNormal(mean=0,stddev=1),
                                  trainable=True)

    def call(self,x,training=False):
        '''
        x:(batch,h,w,c)
        '''
        mean=tf.reduce_mean(x,keepdims=True,axis=[1,2]) #batch,1,1,c
        std=tf.math.reduce_std(x,keepdims=True,axis=[1,2]) #batch,1,1,c
        z=(x-mean)/(std+self.ep)
        return self.gamma*z+self.beta




class ResBlock(Layer):
    def __init__(self,kernel_size,out_channels,norm='none',activation='relu',padding='valid',strides=1):
        super(ResBlock, self).__init__()
        self.conv1=ConvBlock(norm=norm,out_channels=out_channels,kernel_size=kernel_size,activation=activation,padding=padding,strides=strides)
        self.conv2 = ConvBlock(norm=norm,out_channels=out_channels, kernel_size=kernel_size, activation='none', padding=padding,strides=strides)

        if activation=='relu':
            self.activation=ReLU()
        elif activation=='leaky':
            self.activation=LeakyReLU(alpha=0.2)

    def call(self,x,training=False):
        out1=self.conv1(x)
        out2=self.conv2(out1)
        out2=out2+x
        return self.activation(out2)



class ConvBlock(Layer):
    def __init__(self,kernel_size,out_channels,norm='none',activation=None,padding='valid',strides=1):
        super(ConvBlock,self).__init__()

        self.padding=padding
        #padding method
        if padding=='valid':
            self.conv=Conv2D(filters=out_channels,kernel_size=kernel_size,activation=None,padding=padding,strides=strides)
        elif padding=='zeros':
            self.conv = Conv2D(filters=out_channels, kernel_size=kernel_size, activation=None, padding=padding,strides=strides)
        elif padding=='reflect':
            self.reflect=ReflectionPadding2D()
            self.conv = Conv2D(filters=out_channels, kernel_size=kernel_size, activation=None, padding='valid',strides=strides)
        elif padding=='reflect_2':
            self.reflect=ReflectionPadding2D((2,2))
            self.conv = Conv2D(filters=out_channels, kernel_size=kernel_size, activation=None, padding='valid',strides=strides)
        elif padding=='reflect_3':
            self.reflect=ReflectionPadding2D((3,3))
            self.conv = Conv2D(filters=out_channels, kernel_size=kernel_size, activation=None, padding='valid',strides=strides)

        #normalization method
        if norm=='in':
            self.norm=Instance_norm()
        elif norm=='adain':
            self.norm=AdaIN(out_channels)
        elif norm=='ln':
            self.norm=LayerNormalization()
        else:
            self.norm=None

        #activation
        if activation=='relu':
            self.activation=ReLU()
        elif activation=='leaky':
            self.activation=LeakyReLU(alpha=0.2)
        elif activation=='tanh':
            self.activation=Activation(activation=tf.nn.tanh)
        else:
            self.activation=None

    def call(self,x,training=False):
        if 'reflect' in self.padding:
            x=self.reflect(x)
        x=self.conv(x)
        if self.norm!=None:
            x=self.norm(x)
        if self.activation!=None:
            x=self.activation(x)
        return x

'''
x=tf.random.normal((4,32,32,3))
c=ConvBlock(kernel_size=3,out_channels=32,norm='in',activation='relu',padding='reflect')
print(c(x).shape)
x=tf.random.normal((4,32,32,8))
res=ResBlock(kernel_size=3,out_channels=8,norm='in',activation='relu',padding='reflect')
print(res(x).shape)
print(res.conv1.norm.gamma.trainable)
'''

class Content_Enc(Layer):
    def __init__(self,activation='relu',n_res=4):
        super(Content_Enc,self).__init__()
        self.conv=tf.keras.Sequential([
            ConvBlock(kernel_size=7, out_channels=64, norm='in', activation=activation, padding='reflect_3'),
            ConvBlock(kernel_size=4, out_channels=128, norm='in', activation=activation, padding='reflect',strides=2),
            ConvBlock(kernel_size=4, out_channels=256, norm='in', activation=activation, padding='reflect', strides=2),
        ])

        self.res=tf.keras.Sequential([ResBlock(kernel_size=3,out_channels=256,norm='in',activation=activation,padding='reflect')  for _ in range(n_res)])
    def call(self,x,training=False):
        x=self.conv(x)
        x=self.res(x)
        return x

'''
c=Content_Enc()
x=tf.random.normal((5,32,32,3))
print(c(x).shape)
'''


class Style_Enc(Layer):
    def __init__(self,activation='relu'):
        super(Style_Enc,self).__init__()
        self.conv=tf.keras.Sequential([
            ConvBlock(kernel_size=7, out_channels=64, norm='none', activation=activation, padding='reflect_3'),
            ConvBlock(kernel_size=4, out_channels=128, norm='none', activation=activation, padding='reflect',strides=2),
            ConvBlock(kernel_size=4, out_channels=256, norm='none', activation=activation, padding='reflect', strides=2),
            ConvBlock(kernel_size=4, out_channels=256, norm='none', activation=activation, padding='reflect', strides=2),
            ConvBlock(kernel_size=4, out_channels=256, norm='none', activation=activation, padding='reflect', strides=2)
        ])
        self.gp=GlobalAvgPool2D() #just mean over spatial domain
        self.out_feature_map=Conv2D(kernel_size=1,filters=8)


    def call(self,x,training=False):
        x=self.conv(x)
        x=self.gp(x)[:,tf.newaxis,tf.newaxis,:]
        x=self.out_feature_map(x)
        return x
'''
x=tf.random.normal((4,256,256,3))
s=Style_Enc()
print(s(x).shape)
'''

class Decoder(Layer):
    def __init__(self,activation='relu',n_res=4):
        super(Decoder,self).__init__()
        self.res = tf.keras.Sequential([ResBlock(kernel_size=3, out_channels=256, norm='adain', activation=activation, padding='reflect') for _ in range(n_res)])
        self.upsample=tf.keras.Sequential([tf.keras.Sequential([
            UpSampling2D(size=(2,2)),
            ConvBlock(kernel_size=5,out_channels=int(256//(2**(i+1))),norm='ln',activation=activation,padding='reflect_2')
        ]) for i in range(2)])
        self.out_feature_map=ConvBlock(kernel_size=7,out_channels=3,norm='none',activation='tanh',padding='reflect_3')

    def call(self,x,training=False):
        x=self.res(x)
        x=self.upsample(x,training=training)
        x=self.out_feature_map(x)
        return x

'''
x=tf.random.normal((6,32,32,3))
e=Content_Enc()
d=Decoder()
print(d(e(x)).shape)
'''


