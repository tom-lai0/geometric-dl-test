# dl-test

## Introduction
  This program tries to perform a dimension reduction from a 3d object to a 2d plane.
  At the point, the 3d object is an unit sphere.

  The loss functions used are kl divergence, reconstruction error and exponential inverse distance.
  The exponential inverse distance measures the difference between pairwise distance of original points and pairwise distance of encoded points.

## Experiment Results

### Figures

- # 1. Mid-way Figures

    ![midway](https://github.com/tom-lai0/dl-test/blob/main/experiment_1/encoded_epoch_500.png)
    
    Images with name 'encoded_epoch_xxxx.png'.
    The images plots the encoded latent space during training at epoch no. xxxx.

    The colors in the images represent the original distribution.
    For example, the rightmost plot with title 'z' in the above represent the original z-coordinate.
    The original z-coordinate of yellow points are closed to 1, while the original z-coordinate of purple points are closed to -1.
    Similarly for the other two plots with titles 'x' or 'y'.
    A desired model should give plots that the colors of all 3 plots are well-separated.


- # 2. Final Latent Space Figures

    ![final](https://github.com/tom-lai0/dl-test/blob/main/experiment_1/encoded.png)

    Images with name 'encoded.png'.
    The images plots the encoded latent space after training.
    The color have the same meaning as the mid-way figures.
    A desired model should give plots that the colors of all 3 plots are well-separated.

- # 3. Loss History Plots

    ![loss-hist](https://github.com/tom-lai0/dl-test/blob/main/experiment_1/loss_history.png)

    Images with name 'loss_history.png', plot the history of the three loss values.

- # 4. Projection of the Original Point Cloud
    
    ![origin-proj](https://github.com/tom-lai0/dl-test/blob/main/experiment_1/original.png)

    Images with name 'original.png'.
    The images plot the projections of the original 3d point cloud to 2d plane.
    For example, the leftmost plot with title 'z' plots the x-coordinate and y-coordinate of the original point cloud. And the colors of point reflect the z-coordinate. Yellow points have z-coordinate closed to 1, while purple points have z-coordinate closed to -1.

- # 5. Projection of the Reconstructed Point Cloud

    ![recon-proj](https://github.com/tom-lai0/dl-test/blob/main/experiment_1/reconstruction.png)

    Images with name 'reconstruction.png'.
    The images plot the projections of the reconstructed 3d point cloud to 2d plane.
    The meaning of the plots are similar to (4). 
    If the model is well-trained with little reconstruction error, this images should look similar to (4).



    
    



