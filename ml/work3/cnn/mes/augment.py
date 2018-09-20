#!/usr/bin/python

import Augmentor

p = Augmentor.Pipeline("./data/")
p.random_distortion(probability=1, grid_width=5, grid_height=5, magnitude=5)
p.sample(20000)
p.process()