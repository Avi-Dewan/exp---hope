''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os

import gan_trainer.gan_utils as utils
import gan_trainer.gan_loss as gan_loss


# Dummy training function for debugging
def dummy_training_function():
  def train(x, y):
    return {}
  return train


def GAN_training_function(G, D, GD, z_, y_, batch_size, num_D_steps, num_D_accumulations, num_G_accumulations):

  def train(x, y):
    
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, batch_size) 
    y = torch.split(y, batch_size)
    counter = 0
    
    
    utils.toggle_grad(D, True)
    utils.toggle_grad(G, False)
      
    for step_index in range(num_D_steps):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(num_D_accumulations):
        z_.sample_()
        y_.sample_()
        D_fake, D_real = GD(z_[:batch_size], y_[:batch_size], 
                            x[counter], y[counter], train_G=False)
         
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = gan_loss.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(num_D_accumulations)
        D_loss.backward()
        counter += 1
        
      
      D.optim.step()
    
  
    utils.toggle_grad(D, False)
    utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(num_G_accumulations):    
      z_.sample_()
      y_.sample_()
      D_fake = GD(z_, y_, train_G=True)
      G_loss = gan_loss.generator_loss(D_fake) / float(num_G_accumulations)
      G_loss.backward()
    
    
    G.optim.step()
    
    
    out = {'G_loss': float(G_loss.item()), 
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    return out
  return train
  
''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                    state_dict, config, experiment_name):
  utils.save_weights(G, D, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None)
  # Save an additional copy to mitigate accidental corruption if process
  # is killed during a save (it's happened to me before -.-)
  if config['num_save_copies'] > 0:
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'copy%d' %  state_dict['save_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
    
  # Use EMA G for samples or non-EMA?
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Accumulate standing statistics?
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  
  # Save a random sample sheet with fixed z and y      
  with torch.no_grad():
    if config['parallel']:
      fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
    else:
      fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
  if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
    os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'], 
                                                  experiment_name,
                                                  state_dict['itr'])
  torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                             nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
  print('Saved %s' % image_filename)
  # For now, every time we save, also save sample sheets
  utils.sample_sheet(which_G,
                     classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                     num_classes=config['n_classes'],
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'],
                     z_=z_)
  # Also save interp sheets
  for fix_z, fix_y in zip([False, False, True], [False, True, False]):
    utils.interp_sheet(which_G,
                       num_per_sheet=16,
                       num_midpoints=8,
                       num_classes=config['n_classes'],
                       parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       sheet_number=0,
                       fix_z=fix_z, fix_y=fix_y, device='cuda')


  
''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''
def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
  print('Gathering inception metrics...')
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  IS_mean, IS_std, FID = get_inception_metrics(sample, 
                                               config['num_inception_images'],
                                               num_splits=10)
  print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
  # If improved over previous best metric, save approrpiate copy
  if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
    or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
    print('%s improved over previous best, saving checkpoint...' % config['which_best'])
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best%d' % state_dict['save_best_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
  state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
  state_dict['best_FID'] = min(state_dict['best_FID'], FID)
  # Log results to file
  test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
               IS_std=float(IS_std), FID=float(FID))