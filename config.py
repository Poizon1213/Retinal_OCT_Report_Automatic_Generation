import torch
import torch.nn as nn

min_word_feq = 3
train_filename = './data/part_data/traindata.txt'
val_filename = './data/part_data/valdata.txt'
test_filename = './data/part_data/valdata.txt'

image_folder =
data_folder = './data/part_data'
max_len = 70
batch_size = 8
#encoder_dim = 768
encoder_dim = 1024
teacher_forcing_ratio = 0.5
# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 1e-4  # learning rate for decoder
start_epoch = 0
epochs = 100  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 30  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none
# checkpoint = 'checkpoint_.pth.tar'
best_checkpoint = './save_model/BEST_checkpoint_.pth.tar'
workers = 0
grad_clip = 5.  # clip gradients at an absolute value of

bce = nn.BCELoss().to(device)
















