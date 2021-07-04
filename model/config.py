gan_w=1                      # weight of adversarial loss
recon_x_w= 10                 # weight of image reconstruction loss
recon_s_w=1                  # weight of style reconstruction loss
recon_c_w=1                  # weight of content reconstruction loss
recon_x_cyc_w=10

beta1= 0.5                    # Adam parameter
beta2=0.999
lr=0.0001                    # initial learning rate
batch_size=1
epochs=20
image_display_epoch=1
num_gen_img=10

style_code_dim=8


image_size=128
load_model=False
train_x_pth='../data/train/cat/'
train_y_pth='../data/train/face/'
val_x_pth='../data/val/cat/'
val_y_pth='../data/val/face/'
save_pth='./save/train'
res_pth='./res'
shuffle=1000