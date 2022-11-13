import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math
import copy
import random

#############################################################
def DataTransform(sample, config):
   '''
   If you want to change augmentation method, you should change function(scaling and jitter).
   - Default
       - scaling
       - jitter
       - permutation
   - Standard deviation * 2
       - scaling_mul
       - jitter_mul
       - permutation_mul
   - Standard deviation / 2
       - scaling_div
       - jitter_div
       - permutation_div
   - Filp(reverse ver.)
       - scaling_filp_reverse
   + Cropping in this function. If you use this, use the comments section at the bottom. It may take some time to execute this.
   + Filp(negative ver.), Shuffle, Sampling, Spike in this function. If you use this, use the comments section at the bottom.
   '''
   weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
   strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

   # <Cropping in strong_aug>
   '''
   N = len(strong_aug)
   mi = random.randrange(0, N)
   ma = random.randrange(0, N)
   if mi > ma:
       mi, ma = ma, mi
   # Cropping
   for i in range(mi, ma+1):
       for j in range(len(strong_aug[i])):
           for k in range(len(strong_aug[i][j])):
               strong_aug[i][j][k] = 0
   '''

   # <Flip negative version>
   # Warning: Very slow..
   '''
   for i in range(len(sample)):
       for j in range(len(sample[i])):
           for k in range(len(sample[i][j])):
               weak_aug[i][j][k] = -weak_aug[i][j][k]
               #strong_aug[i][j][k] = -strong_aug[i][j][k]
   '''
  
   # Shuffle
   '''
   for i in range(len(sample)):
       for j in range(len(sample[i])):
           l = len(sample[i][j])//2
           for k in range(l):
               # swap
               weak_aug[i][j][k], weak_aug[i][j][l] = weak_aug[i][j][l], weak_aug[i][j][k]
               l += 1
   '''

   # Sampling
   '''
   weak_aug_down = deepcopy(weak_aug)
   for i in range(len(sample)):
       for j in range(len(sample[i])):
           # Down-Sampling
           for k in range(0, len(sample[i][j]), 2):
               weak_aug_down[i][j][k] = max(weak_aug[i][j][k], weak_aug[i][j][k+1])
           # Up-Sampling
           for k in range(1, len(sample[i][j])-1, 2):
               weak_aug_down[i][j][k] = (weak_aug_down[i][j][k-1] + weak_aug_down[i][j][k+1])/2
   weak_aug = np.array(weak_aug_down).reshape(len(sample), len(sample[0]), len(sample[0][0]))
   '''

   # Spike
   '''
   for i in range(len(sample)):
       for j in range(len(sample[i])):
           sub_max = max(weak_aug[i][j])
           sub_min = min(weak_aug[i][j])
           sub_avg = sum(weak_aug[i][j]) / len(sample[i][j])
           for k in range(len(sample[i][j])):
               if (k+1) % 16 == 0:
                   if weak_aug[i][j][k] > 0:
                       temp = random.uniform(sub_avg, sub_max)
                       weak_aug[i][j][k] += temp
                   else:
                       temp = random.uniform(sub_avg, -sub_min)
                       weak_aug[i][j][k] -= temp
   '''

   # Step-like Trand
   '''
   for i in range(len(sample)):
       for j in range(len(sample[i])):
           ran = len(sample[i][j]) // 10
          
           num_init = sum(weak_aug[i][j]) / len(weak_aug[i][j])
           num = 0
           cnt = 0
           for k in range(len(sample[i][j])):
               weak_aug[i][j][k] += (num_init * num)
               cnt += 1
               if cnt == ran:
                   cnt = 0
                   num += 1
   '''

   return weak_aug, strong_aug

'''
Default
'''
# strong augmentation(default)
def jitter(x, sigma=0.8):
   # https://arxiv.org/pdf/1706.00527.pdfs
   return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

# weak augmentation(default)
def scaling(x, sigma=1.1):
   # https://arxiv.org/pdf/1706.00527.pdf
   factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
   ai = []
   for i in range(x.shape[1]):
       xi = x[:, i, :]
       ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
   return np.concatenate((ai), axis=1)

# permutation with string augmentation(default)
def permutation(x, max_segments=5, seg_mode="random"):
   orig_steps = np.arange(x.shape[2])

   num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

   ret = np.zeros_like(x)
   for i, pat in enumerate(x):
       if num_segs[i] > 1:
           if seg_mode == "random":
               split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
               split_points.sort()
               splits = np.split(orig_steps, split_points)
           else:
               splits = np.array_split(orig_steps, num_segs[i])
               warp = np.concatenate(np.random.permutation(splits)).ravel()
               ret[i] = pat[0,warp]
       else:
           ret[i] = pat
   return torch.from_numpy(ret)

'''
Standard deviation * 2
'''
# strong augmentation(*2)
def jitter_mul(x, sigma=0.8):
   return x + np.random.normal(loc=0., scale=sigma*2, size=x.shape)
# weak augmentation(*2)
def scaling_mul(x, sigma=1.1):
   factor = np.random.normal(loc=2., scale=sigma*2, size=(x.shape[0], x.shape[2]))
   ai = []
   for i in range(x.shape[1]):
       xi = x[:, i, :]
       ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
   return np.concatenate((ai), axis=1)
# permutation with string augmentation(*2)
def permutation_mul(x, max_segments=5, seg_mode="random"):
   orig_steps = np.arange(x.shape[2])

   num_segs = np.random.randint(1, max_segments*2, size=(x.shape[0]))

   ret = np.zeros_like(x)
   for i, pat in enumerate(x):
       if num_segs[i] > 1:
           if seg_mode == "random":
               split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
               split_points.sort()
               splits = np.split(orig_steps, split_points)
           else:
               splits = np.array_split(orig_steps, num_segs[i])
           warp = np.concatenate(np.random.permutation(splits)).ravel()
           ret[i] = pat[0,warp]
       else:
           ret[i] = pat
   return torch.from_numpy(ret)

'''
Standard deviation / 2
'''
# strong augmentation(/2)
def jitter_div(x, sigma=0.8):
   return x + np.random.normal(loc=0., scale=sigma/2, size=x.shape)
# weak augmentation(/2)
def scaling_div(x, sigma=1.1):
   factor = np.random.normal(loc=2., scale=sigma/2, size=(x.shape[0], x.shape[2]))
   ai = []
   for i in range(x.shape[1]):
       xi = x[:, i, :]
       ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
   return np.concatenate((ai), axis=1)
# permutation with string augmentation(/2)
def permutation_div(x, max_segments=5, seg_mode="random"):
   orig_steps = np.arange(x.shape[2])

   num_segs = np.random.randint(1, max_segments/2, size=(x.shape[0]))

   ret = np.zeros_like(x)
   for i, pat in enumerate(x):
       if num_segs[i] > 1:
           if seg_mode == "random":
               split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
               split_points.sort()
               splits = np.split(orig_steps, split_points)
           else:
               splits = np.array_split(orig_steps, num_segs[i])
           warp = np.concatenate(np.random.permutation(splits)).ravel()
           ret[i] = pat[0,warp]
       else:
           ret[i] = pat
   return torch.from_numpy(ret)

'''
Filp(reverse ver.)
'''
# weak augmentation(filp_reverse)
def scaling_filp_reverse(x, sigma=1.1):
   factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
   ai = []
   for i in range(x.shape[1]):
       xi = x[:, i, :]
       ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
   ai.reverse()
   return np.concatenate((ai), axis=1)
#############################################################

class TS2Vec:
   '''The TS2Vec model'''
  
   def __init__(
       self,
       input_dims,
       output_dims=320,
       hidden_dims=64,
       depth=10,
       device='cuda',
       lr=0.001,
       batch_size=16,
       max_train_length=None,
       temporal_unit=0,
       after_iter_callback=None,
       after_epoch_callback=None
   ):
       ''' Initialize a TS2Vec model.
      
       Args:
           input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
           output_dims (int): The representation dimension.
           hidden_dims (int): The hidden dimension of the encoder.
           depth (int): The number of hidden residual blocks in the encoder.
           device (int): The gpu used for training and inference.
           lr (int): The learning rate.
           batch_size (int): The batch size.
           max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
           temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
           after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
           after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
       '''
      
       super().__init__()
       self.device = device
       self.lr = lr
       self.batch_size = batch_size
       self.max_train_length = max_train_length
       self.temporal_unit = temporal_unit
      
       self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
       self.net = torch.optim.swa_utils.AveragedModel(self._net)
       self.net.update_parameters(self._net)
      
       self.after_iter_callback = after_iter_callback
       self.after_epoch_callback = after_epoch_callback
      
       self.n_epochs = 0
       self.n_iters = 0
  
   def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
       ''' Training the TS2Vec model.
      
       Args:
           train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
           n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
           n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
           verbose (bool): Whether to print the training loss after each epoch.
          
       Returns:
           loss_log: a list containing the training losses on each epoch.
       '''
       assert train_data.ndim == 3
      
       if n_iters is None and n_epochs is None:
           n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
      
       if self.max_train_length is not None:
           sections = train_data.shape[1] // self.max_train_length
           if sections >= 2:
               train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

       temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
       if temporal_missing[0] or temporal_missing[-1]:
           train_data = centerize_vary_length_series(train_data)

       train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
       #train_data = jitter(train_data[~np.isnan(train_data).all(axis=2).all(axis=1)], 0.2)
       #############################################################
       # train_data = jitter(train_data, 0.2)

       #train_data = scaling(train_data, 1.1)
  
       # train_data = permutation(train_data, 5, "random")
      
       # Down-Up Sampling
       # data_down = copy.deepcopy(train_data)
       # for i in range(len(train_data)):
       #     # Down-Sampling
       #     for j in range(0, len(train_data[i])-1, 2):
       #         data_down[i][j][0] = max(train_data[i][j][0], train_data[i][j+1][0])
       #     # Up-Sampling
       #     for j in range(1, len(train_data[i])-2, 2):
       #         data_down[i][j][0] = (data_down[i][j-1][0] + data_down[i][j+1][0])/2
       # train_data = data_down

       # Spike
       # for i in range(len(train_data)):
       #     for j in range(len(train_data[i])):
       #         sub_max = max(train_data[i][j])
       #         sub_min = min(train_data[i][j])
       #         sub_avg = sum(train_data[i][j]) / len(train_data[i][j])
       #         for k in range(len(train_data[i][j])):
       #             if (k+1) % 16 == 0:
       #                 if train_data[i][j][0] > 0:
       #                     temp = np.random.uniform(sub_avg, sub_max)
       #                     train_data[i][j][0] += temp
       #                 else:
       #                     temp = np.random.uniform(sub_avg, -sub_min)
       #                     train_data[i][j][0] -= temp

       # Shuffle
       # for i in range(len(train_data)):
       #     for j in range(len(train_data[i])):
       #         l = len(train_data[i][j])//2
       #         for k in range(l):
       #             # swap
       #             train_data[i][j][k], train_data[i][j][l] = train_data[i][j][l], train_data[i][j][k]
       #             l += 1
      
       # Step-like Trand
       # for i in range(len(train_data)):
       #     num_init = 0
       #     num_cnt = 0
       #     for j in range(len(train_data[i])):
       #         num_init = max(num_init, train_data[i][j][0])
       #         num_cnt += 1
              
       #     num = 0
       #     cnt = 0
       #     for j in range(0, len(train_data[i]), 10):
       #         for k in range(10):
       #             train_data[i][j+k][0] += (num_init * num)
       #             cnt += 1
       #             if cnt >= num_cnt:
       #                 break
       #         num += 1

       # Flip
       # for i in range(len(train_data)):
       #     for j in range(len(train_data[i])):
       #         for k in range(len(train_data[i][j])):
       #             train_data[i][j][k] = -train_data[i][j][k]
       #############################################################

       train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))

       train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
       optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
      
       loss_log = []

       while True:
           if n_epochs is not None and self.n_epochs >= n_epochs:
               break
          
           cum_loss = 0
           n_epoch_iters = 0
          
           interrupted = False
           for batch in train_loader:
               if n_iters is not None and self.n_iters >= n_iters:
                   interrupted = True
                   break

               x = batch[0]

               if self.max_train_length is not None and x.size(1) > self.max_train_length:
                   window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                   x = x[:, window_offset : window_offset + self.max_train_length]
               #####################
               # Data Augmentation #
               #####################
               # def jitter_(x):
               #     return x + torch.normal(0, 0.2, x.shape).cuda()
              
               # x = jitter_(x.to(self.device))
               x = x.to(self.device)

               ts_l = x.size(1)
               crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
               crop_left = np.random.randint(ts_l - crop_l + 1)
               crop_right = crop_left + crop_l
               crop_eleft = np.random.randint(crop_left + 1)
               crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
               crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
              
               optimizer.zero_grad()
              
               out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
               out1 = out1[:, -crop_l:]

               out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
               out2 = out2[:, :crop_l]
              
               #####################
               # Data Augmentation #
               #####################
               # def jitter_(x):
               #     return x + torch.normal(0, 0.2, x.shape).cuda()
               # def scaling_(x):
               #     factor = torch.normal(2, 1.1, (x.shape[0], x.shape[2])).cuda()
               #     ai = []
               #     for i in range(x.shape[1]):
               #         xi = x[:, i, :]
               #         ai.append(torch.multiply(xi, factor[:, :])[:, np.newaxis, :])
               #     return torch.cat((ai), axis=1)
               #
               # out1 = jitter_(out1)
               # out2 = jitter_(out1)
               #
               # out1 = jitter_(out1)
               # out2 = scaling_(out2)

               loss = hierarchical_contrastive_loss(
                   out1,
                   out2,
                   temporal_unit=self.temporal_unit
               )
              
               loss.backward()
               optimizer.step()
               self.net.update_parameters(self._net)
                  
               cum_loss += loss.item()
               n_epoch_iters += 1
              
               self.n_iters += 1
              
               if self.after_iter_callback is not None:
                   self.after_iter_callback(self, loss.item())
          
           if interrupted:
               break
          
           cum_loss /= n_epoch_iters
           loss_log.append(cum_loss)
           if verbose:
               print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
           self.n_epochs += 1
          
           if self.after_epoch_callback is not None:
               self.after_epoch_callback(self, cum_loss)
          
       return loss_log
  
   def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
       out = self.net(x.to(self.device, non_blocking=True), mask)
       if encoding_window == 'full_series':
           if slicing is not None:
               out = out[:, slicing]
           out = F.max_pool1d(
               out.transpose(1, 2),
               kernel_size = out.size(1),
           ).transpose(1, 2)
          
       elif isinstance(encoding_window, int):
           out = F.max_pool1d(
               out.transpose(1, 2),
               kernel_size = encoding_window,
               stride = 1,
               padding = encoding_window // 2
           ).transpose(1, 2)
           if encoding_window % 2 == 0:
               out = out[:, :-1]
           if slicing is not None:
               out = out[:, slicing]
          
       elif encoding_window == 'multiscale':
           p = 0
           reprs = []
           while (1 << p) + 1 < out.size(1):
               t_out = F.max_pool1d(
                   out.transpose(1, 2),
                   kernel_size = (1 << (p + 1)) + 1,
                   stride = 1,
                   padding = 1 << p
               ).transpose(1, 2)
               if slicing is not None:
                   t_out = t_out[:, slicing]
               reprs.append(t_out)
               p += 1
           out = torch.cat(reprs, dim=-1)
          
       else:
           if slicing is not None:
               out = out[:, slicing]
          
       return out.cpu()
  
   def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
       ''' Compute representations using the model.
      
       Args:
           data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
           mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
           encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
           casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
           sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
           sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
           batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
          
       Returns:
           repr: The representations for data.
       '''
       assert self.net is not None, 'please train or load a net first'
       assert data.ndim == 3
       if batch_size is None:
           batch_size = self.batch_size
       n_samples, ts_l, _ = data.shape

       org_training = self.net.training
       self.net.eval()
      
       dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
       loader = DataLoader(dataset, batch_size=batch_size)
      
       with torch.no_grad():
           output = []
           for batch in loader:
               x = batch[0]
               if sliding_length is not None:
                   reprs = []
                   if n_samples < batch_size:
                       calc_buffer = []
                       calc_buffer_l = 0
                   for i in range(0, ts_l, sliding_length):
                       l = i - sliding_padding
                       r = i + sliding_length + (sliding_padding if not casual else 0)
                       x_sliding = torch_pad_nan(
                           x[:, max(l, 0) : min(r, ts_l)],
                           left=-l if l<0 else 0,
                           right=r-ts_l if r>ts_l else 0,
                           dim=1
                       )
                       if n_samples < batch_size:
                           if calc_buffer_l + n_samples > batch_size:
                               out = self._eval_with_pooling(
                                   torch.cat(calc_buffer, dim=0),
                                   mask,
                                   slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                   encoding_window=encoding_window
                               )
                               reprs += torch.split(out, n_samples)
                               calc_buffer = []
                               calc_buffer_l = 0
                           calc_buffer.append(x_sliding)
                           calc_buffer_l += n_samples
                       else:
                           out = self._eval_with_pooling(
                               x_sliding,
                               mask,
                               slicing=slice(sliding_padding, sliding_padding+sliding_length),
                               encoding_window=encoding_window
                           )
                           reprs.append(out)

                   if n_samples < batch_size:
                       if calc_buffer_l > 0:
                           out = self._eval_with_pooling(
                               torch.cat(calc_buffer, dim=0),
                               mask,
                               slicing=slice(sliding_padding, sliding_padding+sliding_length),
                               encoding_window=encoding_window
                           )
                           reprs += torch.split(out, n_samples)
                           calc_buffer = []
                           calc_buffer_l = 0
                  
                   out = torch.cat(reprs, dim=1)
                   if encoding_window == 'full_series':
                       out = F.max_pool1d(
                           out.transpose(1, 2).contiguous(),
                           kernel_size = out.size(1),
                       ).squeeze(1)
               else:
                   out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                   if encoding_window == 'full_series':
                       out = out.squeeze(1)
                      
               output.append(out)
              
           output = torch.cat(output, dim=0)
          
       self.net.train(org_training)
       return output.numpy()
  
   def save(self, fn):
       ''' Save the model to a file.
      
       Args:
           fn (str): filename.
       '''
       torch.save(self.net.state_dict(), fn)
  
   def load(self, fn):
       ''' Load the model from a file.
      
       Args:
           fn (str): filename.
       '''
       state_dict = torch.load(fn, map_location=self.device)
       self.net.load_state_dict(state_dict)
  


