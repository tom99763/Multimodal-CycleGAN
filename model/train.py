import tensorflow as tf
from model import Generator,Discriminator
import config
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import cv2

aug = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x:tf.image.random_flip_left_right(x)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5,offset=-1)
])

def get_data(type):
    ds_X=tf.keras.preprocessing.image_dataset_from_directory(
        directory=config.train_x_pth if type=='train' else config.val_x_pth,
        shuffle=False,
        image_size=(config.image_size,config.image_size),
        batch_size=1
    )

    ds_Y=tf.keras.preprocessing.image_dataset_from_directory(
        directory=config.train_y_pth if type=='train' else config.val_y_pth,
        shuffle=False,
        image_size=(config.image_size,config.image_size),
        batch_size=1
    )

    ds=tf.data.Dataset.zip((ds_X,ds_Y)).shuffle(config.shuffle).batch(config.batch_size,drop_remainder=False).\
        cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def gen_val_img(ds, Gx, Gy):
    for i, (x, y) in enumerate(ds):
        if i == config.num_gen_img:
            return
        x, y = x[0][:, 0, ...], y[0][:, 0, ...]
        x, y = aug(x), aug(y)

        cx,sx=Gx.encode(x,training=False)
        cy,sy=Gy.encode(y,training=False)

        x_hat=Gx.decode(cy,sx,training=False)[0, ...] * 127.5 + 127.5
        y_hat=Gy.decode(cx,sy,training=False)[0, ...] * 127.5 + 127.5

        y_hat = y_hat.numpy().astype('uint8')
        x_hat = x_hat.numpy().astype('uint8')
        y_hat = cv2.cvtColor(y_hat, cv2.COLOR_BGR2RGB)
        x_hat = cv2.cvtColor(x_hat, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f'{config.res_pth}/X2Y_{i}.jpg', y_hat)
        cv2.imwrite(f'./{config.res_pth}/Y2X_{i}.jpg', x_hat)

def flow(x,y,Gx,Gy,Dx,Dy,Gx_opt,Gy_opt,Dx_opt,Dy_opt):
    l1=tf.keras.losses.MeanAbsoluteError()
    l2=tf.keras.losses.MeanSquaredError()
    with tf.GradientTape(persistent=True) as tape:
        #flow
        sx=tf.random.normal((config.batch_size,1,1,config.style_code_dim))
        sy=tf.random.normal((config.batch_size,1,1,config.style_code_dim))

        cx,sx_prime=Gx.encode(x,training=True)
        cy,sy_prime=Gy.encode(y,training=True)
        x_recon=Gx.decode(cx,sx_prime,training=True)
        y_recon=Gy.decode(cy,sy_prime,training=True)

        y_sx=Gx.decode(cy,sx,training=True)
        x_sy=Gy.decode(cx,sy,training=True)

        cy_hat,sx_hat=Gx.encode(y_sx,training=True)
        cx_hat,sy_hat=Gy.encode(x_sy,training=True)

        x_hat=Gx.decode(cx,sx_hat,training=True)
        y_hat=Gy.decode(cy,sy_hat,training=True)

        #self recontruction error(autoencoder)
        self_recon_error_x=l1(x,x_recon)
        self_recon_error_y=l1(y,y_recon)

        #content code recontruction error
        content_recon_error_x=l1(cx,cx_hat)
        content_recon_error_y=l1(cy,cy_hat)

        #style code recontruction error
        style_recon_error_x=l1(sx,sx_hat)
        style_recon_error_y=l1(sy,sy_hat)

        #style-aug cycle-consistencty
        x_cycle_loss=l1(x,x_hat)
        y_cycle_loss=l1(y,y_hat)

        #GAN Loss(least squared GAN)
        dx_loss=0
        dy_loss=0
        g_loss=0
        dx_out_real,dx_out_fake=Dx(x,training=True),Dx(y_sx,training=True)
        dy_out_real,dy_out_fake=Dy(y, training=True), Dy(x_sy, training=True)

        for dx_o_r,dx_o_f,dy_o_r,dy_o_f in zip(dx_out_real,dx_out_fake,dy_out_real,dy_out_fake):
            dx_loss+= l2(1,dx_o_r)+l2(0,dx_o_f)
            dy_loss+= l2(1,dy_o_r)+l2(0,dy_o_f)
            g_loss+=l2(1,dx_o_f)
            g_loss+=l2(1,dy_o_f)

        g_total_loss = config.gan_w * (g_loss) + \
                       config.recon_x_w * (self_recon_error_x + self_recon_error_y) + \
                       config.recon_c_w*(content_recon_error_x + content_recon_error_y) + \
                       config.recon_s_w*(style_recon_error_x + style_recon_error_y) + \
                       config.recon_x_cyc_w*(x_cycle_loss + y_cycle_loss)

        dx_loss*=config.gan_w
        dy_loss*=config.gan_w

    dx_grads=tape.gradient(dx_loss,Dx.trainable_weights)
    dy_grads=tape.gradient(dy_loss,Dy.trainable_weights)
    gx_grads=tape.gradient(g_total_loss,Gx.trainable_weights)
    gy_grads=tape.gradient(g_total_loss,Gy.trainable_weights)

    Gx_opt.apply_gradients(zip(gx_grads,Gx.trainable_weights))
    Gy_opt.apply_gradients(zip(gy_grads, Gy.trainable_weights))
    Dx_opt.apply_gradients(zip(dx_grads, Dx.trainable_weights))
    Dy_opt.apply_gradients(zip(dy_grads, Dy.trainable_weights))
    return g_total_loss,dx_loss,dy_loss



def main():
    Gx_opt=Adam(learning_rate=config.lr,beta_1=config.beta1,beta_2=config.beta2)
    Gy_opt=Adam(learning_rate=config.lr,beta_1=config.beta1,beta_2=config.beta2)
    Dx_opt=Adam(learning_rate=config.lr,beta_1=config.beta1,beta_2=config.beta2)
    Dy_opt=Adam(learning_rate=config.lr,beta_1=config.beta1,beta_2=config.beta2)

    Gx=Generator()
    Gy=Generator()
    Dx=Discriminator()
    Dy=Discriminator()

    ckpt = tf.train.Checkpoint(Dx=Dx, Dy=Dy, Gx=Gx, Gy=Gy, Dx_optim=Dx_opt, Dy_optim=Dy_opt,
                               Gx_optim=Gx_opt, Gy_optim=Gy_opt)
    ckpt_manager = tf.train.CheckpointManager(ckpt, config.save_pth, max_to_keep=1)
    if ckpt_manager.latest_checkpoint and config.load_model:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('---ckpt restored----')
    train_ds=get_data('train')
    val_ds = get_data('val')


    gen_val_img(val_ds, Gx, Gy)

    for epoch in range(config.epochs):
        loop = tqdm(train_ds, leave=True)
        for x,y in loop:
            x,y=x[0][:, 0, ...], y[0][:, 0, ...]
            x, y = aug(x), aug(y)
            g_total_loss,dx_loss,dy_loss=flow(x,y,Gx,Gy,Dx,Dy,Gx_opt,Gy_opt,Dx_opt,Dy_opt)
            loop.set_postfix(
                loss=f'epoch:{epoch} g_total_loss:{g_total_loss},dx_loss:{dx_loss},dy_loss:{dy_loss}')
        ckpt_manager.save()
        if epoch%config.image_display_epoch==0:
            gen_val_img(val_ds,Gx, Gy)

if __name__ == '__main__':
    main()

