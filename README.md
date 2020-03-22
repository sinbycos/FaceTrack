
### Introduction
The algorithm proposed in the paper titled 'Robust Face Tracking using Multiple Appearance Models and Graph Relational Learning' addresses the problem of appearance matching across different challenges while doing visual face tracking in real-world scenarios. 
In this paper, FaceTrack is proposed that utilizes multiple appearance models with its long-term and short-term appearance memory for efficient face tracking. 
It demonstrates robustness to deformation, in-plane and out-of-plane rotation, scale, distractors and background clutter. 
It capitalizes on the advantages of the tracking-by-detection, by using a face detector that tackles drastic scale appearance change of a face. 
The detector also helps to reinitialize FaceTrack during drift. A weighted score-level fusion strategy is proposed to obtain the face tracking output having the highest fusion score by generating candidates around possible face locations. 
The tracker showcases impressive performance when initiated automatically by outperforming many state-of-the-art trackers, except Struck by a very minute margin: 0.001 in precision and 0.017 in success respectively. 


FaceTrack was initially described on [arXiv](https://arxiv.org/abs/1706.09806) and will appear in Machine Vision and Applications as a journal paper.

### Citing FaceTrack

If you find FaceTrack useful in your research, please consider citing:

    @inproceedings{facetrackMVA,
        Author = {Tanushri Chakravorty, Guillaume-Alexandre Bilodeau, Eric Granger},
        Title = {Robust Face Tracking using Multiple Appearance Models and Graph Relational Learning},
        Journal = {Machine Vision and Applications},
        Year = {2020}
    }

### License

FaceTrack is released under the MIT License (refer to the
LICENSE file for details).





