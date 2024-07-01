
Training on a medical dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While we can learn to register 2-D images in a few minutes even on cpu, training for registering 3-D volumes is a more serious endeavor, especially at high resolutions. For that reason, we recommend: 

- Preprocessing all your data in a seperate script and storing it as a :func:`torch.load` / :func:`torch.save` file. This makes loading your dataset fast for iterating changes to your training script, but also prevents you from being bottlenecked by the disk during training.

- Recording all hyperparameters assosciated with each training run so that you can replicate it- this is super important if you are investing hours or days into a training run, and super easy with :mod:`footsteps`

- Generating and saving metrics, visualizations and weight checkpoints throughout training.

Chosing and Downloading a dataset
=================================
For this tutorial we will use the LUMIR dataset and evaluation provided by Learn2Reg 2024. Information on the LUMIR task is hosted on [github](https://github.com/JHU-MedImage-Reg/LUMIR_L2R/). This tutorial assumes you have completed To begin, download the LUMIR data from [Google Drive](https://drive.usercontent.google.com/download?id=1PTHAX9hZX7HBXXUGVvI1ar1LUf4aVbq9&export=download&authuser=0)

.. code-block:: bash
        > ls
        yeeter.zip
        > unzip yeeter.zip
        
 

Preprocessing the Dataset
=========================

.. code-block:: python
   
        import footsteps
        import torch
        import itk
        import tqdm
        import numpy as np
        footsteps.initialize(output_root="/playpen-ssd/tgreer/ICON_brain_preprocessed_data/")

        def process(iA, isSeg=False):
            iA = iA[None, None, :, :, :]
            iA = torch.nn.functional.avg_pool3d(iA, 4)
            return iA

        downsample = 4
        for split in ["train", "test"]:
            with open(f"splits/{split}.txt") as f:
                image_paths = f.readlines()

            ds = []

            for name in tqdm.tqdm(list(iter(image_paths))[:]):
                name = name.split(".nii.gz")[0] + "_restore_brain.nii.gz"

                image = torch.tensor(np.asarray(itk.imread(name)))

                ds.append(process(image))

            torch.save(ds, f"{footsteps.output_dir}/brain_{split}_4xdown_scaled")


This is the script that you most likely need to modify for your own machine and dataset. To run it, we start with a list of filenames for our splits on our buld storage filesystem, load them using itk, downsample them to the resolution we are training at, and then write them as a tensor to our local ssd. This takes close to an hour to run, but means in all subsequent runs we can start training after a few seconds. If your dataset does not fit in RAM (we use a lot of RAM) then this script will need to be more advanced.

Training the Model
==================

Once the data is preprocessed, we train a network to register it. In this example we are doing inter-subject brain registration, so we can just compile batches by sampling random pairs from the dataset. We can use the exact same network architecture from the previous fives example, just setting dimension to 3.

.. code-block:: python

        import random

        import footsteps
        import icon_registration as icon
        import icon_registration.networks as networks
        import torch


        input_shape = [1, 1, 130, 155, 130]

        def make_network():
            inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=2))

            for _ in range(3):
                 inner_net = icon.TwoStepRegistration(
                     icon.DownsampleRegistration(inner_net, dimension=2),
                     icon.FunctionFromVectorField(networks.tallUNet2(dimension=2))
                 )

            net = icon.GradientICON(inner_net, icon.LNCC(sigma=4), lmbda=.5)
            net.assign_identity_map(input_shape)
            return net

We define a custom function for creating and preparing batches of images. Feel free to do this with a torch :class:`torch.Dataset`, but I am more confident about predicting the performance of proceedural code for this task.

.. code-block:: python

        BATCH_SIZE = 8
        GPUS = 4

        def make_batch():
            image = torch.cat([random.choice(brains) for _ in range(GPUS * BATCH_SIZE)])
            image = image.cuda()
            image = image / torch.max(image)
            return image

Then, use the function :func:`icon_registration.train.train_batchfunction` to commence training.

.. code-block:: python

        if __name__ == "__main__":
            footsteps.initialize()
            brains = torch.load(
                "/playpen-ssd/tgreer/ICON_brain_preprocessed_data/stripped/brain_train_2xdown_scaled"
            )
            hires_net = make_network()

            if GPUS == 1:
                net_par = hires_net.cuda()
            else:
                net_par = torch.nn.DataParallel(hires_net).cuda()
            optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

            net_par.train()

            icon.train_batchfunction(net_par, optimizer, lambda: (make_batch(), make_batch()), unwrapped_net=hires_net)
