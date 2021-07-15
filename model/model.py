import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Dense,Flatten,AveragePooling2D
from blocks import Content_Enc,Style_Enc,Decoder,ConvBlock

class Generator(tf.keras.Model):
    def __init__(self,mlp_dim=256,n_res=4):
        super(Generator,self).__init__()
        self.content_encoder=Content_Enc(n_res=n_res)
        self.style_encoder=Style_Enc()
        self.decoder=Decoder(n_res=n_res)
        self.mlp=tf.keras.Sequential([
            Flatten(),
            Dense(units=mlp_dim,activation='relu',kernel_initializer=tf.keras.initializers.HeNormal()),
            Dense(units=mlp_dim, activation='relu',kernel_initializer=tf.keras.initializers.HeNormal()),
            Dense(units=self.get_params_num(),activation=None,kernel_initializer=tf.keras.initializers.HeNormal())
        ])
    def encode(self,x,training=False):
        return self.content_encoder(x,training=training),self.style_encoder(x,training=training)

    def decode(self,c,s,training=False):
        params=self.mlp(s)
        self.assign_params(params)
        return self.decoder(c,training=training)

    def get_params_num(self):
        num=0
        for block in self.decoder.res.layers:
            num+=block.conv1.norm.params_num
            num+=block.conv2.norm.params_num
        return num

    def assign_params(self,params):
        '''
        params:(batch,dim)
        '''
        for block in self.decoder.res.layers:
            n1=block.conv1.norm.params_num
            block.conv1.norm.gamma=params[:,:n1//2][:,tf.newaxis,tf.newaxis,:]
            block.conv1.norm.beta=params[:,n1//2:n1][:,tf.newaxis,tf.newaxis,:]
            n2=block.conv2.norm.params_num
            block.conv2.norm.gamma=params[:,n1:n1+n2//2][:,tf.newaxis,tf.newaxis,:]
            block.conv2.norm.beta=params[:,n1+n2//2:n1+n2][:,tf.newaxis,tf.newaxis,:]
            params=params[:,int(n1+n2):]



class Discriminator(tf.keras.Model):
    def __init__(self,num_scales=3):
        super(Discriminator,self).__init__()
        self.models=[self.net() for _ in range(num_scales)]
        self.downsample=AveragePooling2D()

    def call(self,x,training=False):
        outputs=[]
        for model in self.models:
            outputs.append(model(x,training=training))
            x=self.downsample(x)
        return outputs

    def net(self):
        return tf.keras.Sequential([
            ConvBlock(kernel_size=4, out_channels=64, norm='none', activation='leaky', padding='reflect', strides=2),
            ConvBlock(kernel_size=4, out_channels=128, norm='none', activation='leaky', padding='reflect', strides=2),
            ConvBlock(kernel_size=4, out_channels=256, norm='none', activation='leaky', padding='reflect', strides=2),
            ConvBlock(kernel_size=4, out_channels=512, norm='none', activation='leaky', padding='reflect', strides=2),
            Conv2D(filters=1,kernel_size=1,strides=1,padding='valid',activation=None,kernel_initializer=tf.keras.initializers.HeNormal())
        ])

'''
G=Generator()
D=Discriminator()
x=tf.random.normal((4,256,256,3))
c,s=G.encode(x)
x_hat=G.decode(c,s)
print(x_hat.shape)
print(G.decoder.res.layers[0].conv1.norm.gamma.shape)
out_fake=D(x_hat)
out_real=D(x)
for o,o_prime in zip(out_real,out_fake):
    print(o.shape)
    print(o_prime.shape)
'''



