

Based on this wrapper also create an easy way to evaluate the results on a test set with labels.

Provide and clean up scripts for the training.

 - The exact training scripts used for the paper have been moved into their own folder `training_scripts/oai_paper_pipeline/` (done)

Document. 


## Part 2

For the OAI pipeline we need to support image to atlas-image registration.
This will be different than image-to-image registration as the atlas will
have a slightly different appearance (it is in the repo). Not sure how much
difference that will make. It might require a different similarity measure
(maybe NCC) and/or might require a different kind of training (as not all
the image-pair combinations will be useful, instead it will be
atlas-to-image and image-to-atlas).

Similarity measure: If another similarity than SSD is required (e.g., NCC or
something more sophisticated) is the ICON approach still expected to work?

 - NCC is implemented in [the
   library](https://github.com/HastingsGreer/InverseConsistency/blob/bf488289726e69c70a77ac172f1919e83dc250c9/training_scripts/_/oai_experimental/hires_continue_ramp_lambda.py#L28)
   and works, although with lower final accuracy

Training strategy: Since the goal is image-to-atlas registration there will in
principle be fewer training pairs (as not all pairwise image combinations are
feasible). Will this likely create issues for the approach? I suspect one could
always train an  image-to-image approach first and then fine-tune for
image-to-atlas. Another (maybe more complex alternative) would be to use a
similarity measure as in Zhipeng's paper (attached).

For the registration interface it might make sense to make it a little more
general. E.g., where it would also allow image-segmentations as inputs (if
desired).
