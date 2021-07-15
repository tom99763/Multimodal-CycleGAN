gan_w=1                      # weight of adversarial loss
recon_x_w= 10                 # weight of image reconstruction loss
recon_s_w=1                  # weight of style reconstruction loss
recon_c_w=1                  # weight of content reconstruction loss
recon_x_cyc_w=10

beta1= 0.5                    # Adam parameter
beta2=0.999
lr=0.0001                    # initial learning rate
step_size=100000
gamma=0.5
batch_size=1
epochs=50
image_display=200
num_gen_img= 10
style_code_dim=8
n_dis_scale=3


image_size=256
load_model=False
train_x_pth='../../../data/gan/anime/train/trainA/'
train_y_pth='../../../data/gan/anime/train/trainB/'
val_x_pth='../../../data/gan/anime/test/testA/'
val_y_pth='../../../data/gan/anime/test/testB/'
seed_x=99763
seed_y=33679
save_pth='./save/train'
res_pth='./res'
shuffle=10000


