# TODO:
# - Separate Training and Testing images
# Separate Training and Validation images
# Save an load networks
# Upload results to girder.

# In[1]:

from __future__ import division, print_function
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#import glob
import pdb
plt.rcParams['image.cmap'] = 'gist_earth'


# ## preparing training data
# only one day...

# In[2]:

#get_ipython().system(u'wget -q -r -nH -np --cut-dirs=2 http://people.phys.ethz.ch/~ast/cosmo/bgs_example_data/')


# In[3]:

#get_ipython().system(u'mkdir -p bgs_example_data/seek_cache')


# In[5]:

#get_ipython().system(u"seek --file-prefix='./bgs_example_data' --post-processing-prefix='bgs_example_data/seek_cache' --chi-1=20 --overwrite=True seek.config.process_survey_fft")


### setting up the unet

# In[2]:

from scripts.grayscale_slide_util import GrayscaleSlideDataProvider
from tf_unet import unet


# In[3]:

#files = glob.glob('bgs_example_data/seek_cache/*')


# In[5]:
"""
train_files = [('image2.png','mask2.png'),
               ('image3.png','mask3.png'),
               ('image4.png','mask4.png'),
               ('image5.png','mask5.png'),
               ('image6.png','mask6.png'),
               ('normal1.png','normal_mask1.png'),
               ('normal2.png','normal_mask2.png'),
               ('normal3.png','normal_mask3.png'),
               ('normal4.png','normal_mask4.png'),
               ('normal5.png','normal_mask5.png'),
               ('normal6.png','normal_mask6.png'),
               ('normal7.png','normal_mask7.png'),
               ('normal8.png','normal_mask8.png')]
"""
train_files = [('channel1time0.png','channel1time0_mask.png'),
               ('channel1time1.png','channel1time1_mask.png'),               
               ('channel1time2.png','channel1time2_mask.png'),               
               ('channel1time3.png','channel1time3_mask.png'),               
               ('channel1time4.png','channel1time4_mask.png'),               
               ('channel1time5.png','channel1time5_mask.png'),               
               ('channel1time10.png','channel1time10_mask.png'),
               ('channel1time11.png','channel1time11_mask.png'),               
               ('channel1time12.png','channel1time12_mask.png'),               
               ('channel1time13.png','channel1time13_mask.png'),               
               ('channel1time14.png','channel1time14_mask.png'),               
               ('channel1time15.png','channel1time15_mask.png'),               
               ('channel1time16.png','channel1time16_mask.png'),               
               ('channel1time17.png','channel1time17_mask.png'),               
               ('channel1time18.png','channel1time18_mask.png'),               
               ('channel1time19.png','channel1time19_mask.png'),               
               ('channel1time20.png','channel1time20_mask.png'),
               ('channel1time21.png','channel1time21_mask.png'),               
               ('channel1time22.png','channel1time22_mask.png'),               
               ('channel1time23.png','channel1time23_mask.png'),               
               ('channel1time24.png','channel1time24_mask.png'),               
               ('channel1time25.png','channel1time25_mask.png'),               
               ('channel1time26.png','channel1time26_mask.png'),               
               ('channel1time27.png','channel1time27_mask.png'),               
               ('channel1time28.png','channel1time28_mask.png'),               
               ('channel1time29.png','channel1time29_mask.png'),               
               ('channel1time30.png','channel1time30_mask.png'),
               ('channel1time31.png','channel1time31_mask.png'),               
               ('channel1time32.png','channel1time32_mask.png'),               
               ('channel1time33.png','channel1time33_mask.png'),               
               ('channel1time34.png','channel1time34_mask.png'),               
               ('channel1time35.png','channel1time35_mask.png'),               
               ('channel1time36.png','channel1time36_mask.png'),               
               ('channel1time37.png','channel1time37_mask.png'),               
               ('channel1time38.png','channel1time38_mask.png'),               
               ('channel1time39.png','channel1time39_mask.png'),               
               ('channel1time40.png','channel1time40_mask.png'),
               ('channel1time41.png','channel1time41_mask.png'),               
               ('channel1time42.png','channel1time42_mask.png'),               
               ('channel1time44.png','channel1time44_mask.png'),               
               ('channel1time46.png','channel1time46_mask.png'),               
               ('channel1time47.png','channel1time47_mask.png'),               
               ('channel1time48.png','channel1time48_mask.png'),               
               ('channel1time49.png','channel1time49_mask.png'),               
               ('channel1time50.png','channel1time50_mask.png'),
               ('channel1time51.png','channel1time51_mask.png'),               
               ('channel1time52.png','channel1time52_mask.png'),               
               ('channel1time53.png','channel1time53_mask.png'),               
               ('channel1time54.png','channel1time54_mask.png'),               
               ('channel1time55.png','channel1time55_mask.png'),               
               ('channel1time56.png','channel1time56_mask.png'),               
               ('channel1time57.png','channel1time57_mask.png'),               
               ('channel1time58.png','channel1time58_mask.png'),               
               ('channel1time59.png','channel1time59_mask.png'),               
               ('channel1time60.png','channel1time60_mask.png'),
               ('channel1time61.png','channel1time61_mask.png'),               
               ('channel1time62.png','channel1time62_mask.png'),               
               ('channel1time63.png','channel1time63_mask.png'),               
               ('channel1time64.png','channel1time64_mask.png'),               
               ('channel1time65.png','channel1time65_mask.png'),               
               ('channel1time66.png','channel1time66_mask.png'),               
               ('channel1time67.png','channel1time67_mask.png'),
               ('channel1time68.png','channel1time68_mask.png'),
               ('channel1time69.png','channel1time69_mask.png')]


train_data_provider = GrayscaleSlideDataProvider("/home/preston/data/dna_damage/train", train_files)

net = unet.Unet(channels=train_data_provider.channels, 
                n_class=train_data_provider.n_class, 
                layers=5, 
                features_root=64,
                cost_kwargs=dict(regularizer=0.001),
                filter_size=3
                )


# ## training the network
# only one epoch. For good results many more are neccessary

# In[6]:


trainer = unet.Trainer(net, optimizer="momentum", \
                       opt_kwargs=dict(momentum=0.2, learning_rate=0.2))
# Second argument is output path
# The returned path looks like the file name of a checkpoint.
path = trainer.train(train_data_provider, "./unet_trained_bgs_example_data", 
                     training_iters=16, 
                     epochs=1000, 
                     dropout=0.2, 
                     display_step=1,
                     restore=False)

### running the prediction on the trained unet

# In[10]:
"""
test_files = [('image1.png','mask1.png')]


test_data_provider = SlideDataProvider(test_files, (2000, 2000))
# TODO: Comment returned values: showing the image looks odd.
x_test, y_test = test_data_provider(1)
# TODO: Comment returned value prediction.  I assume this is a mask array.
prediction = net.predict(path, x_test)
#pdb.set_trace()

# TODO: save images

# In[11]:

#fig, ax = plt.subplots(1,3, figsize=(12,4))
#ax[0].imshow(x_test[0], aspect="auto")
#ax[1].imshow(y_test[0,...,1], aspect="auto")
#ax[2].imshow(prediction[0,...,1], aspect="auto")
#plt.show()
#pdb.set_trace()

# In[ ]:
"""


