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

from scripts.slide_util import SlideDataProvider
from tf_unet import unet


# In[3]:

#files = glob.glob('bgs_example_data/seek_cache/*')


# In[5]:
train_files = [('roi1.png','roi1mask.png'), \
               ('roi2.png','roi2mask.png'), \
               ('roi3.png','roi3mask.png')]
train_data_provider = SlideDataProvider(train_files)

net = unet.Unet(channels=train_data_provider.channels, 
                n_class=train_data_provider.n_class, 
                layers=3, 
                features_root=64,
                cost_kwargs=dict(regularizer=0.001),
                )


# ## training the network
# only one epoch. For good results many more are neccessary

# In[6]:


trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
# Second argument is output path
# The returned path looks like the file name of a checkpoint.
path = trainer.train(train_data_provider, "./unet_trained_bgs_example_data", 
                     training_iters=32, 
                     epochs=1, 
                     dropout=0.5, 
                     display_step=2,
                     restore=True)

### running the prediction on the trained unet

# In[10]:

test_files = [('roi4.png','roi4mask.png')]
test_data_provider = SlideDataProvider(test_files, (2000, 2000))
# TODO: Comment returned values: showing the image looks odd.
x_test, y_test = test_data_provider(1)
# TODO: Comment returned value prediction.  I assume this is a mask array.
prediction = net.predict(path, x_test)
#pdb.set_trace()

# TODO: save images

# In[11]:

fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].imshow(x_test[0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
ax[2].imshow(prediction[0,...,1], aspect="auto")
plt.show()
pdb.set_trace()

# In[ ]:



