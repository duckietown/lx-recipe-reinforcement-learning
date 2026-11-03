# LX Recipe: <LX_TITLE_HERE>

TODO: add information that is useful for instructors

# How to Create an LX Recipe

The LX recipe contains components of the exercise that the student should
not need to change and are effectively hidden from view. There are a few
files here that will require some attention in the creation of the learning
experience

## Dependencies

Specify any `apt` or `python` dependencies needed in the 
[dependencies-apt.txt](./dependencies-apt.txt) and 
[dependencies-py3.txt](./dependencies-py3.txt) respectively. 

## Base Image

You need to specify a `BASE_IMAGE` in the [Dockerfile](./Dockerfile). This
choice defines what is already in the docker image and can be used in the 
running of the exercise.

Reasonable choices could be:

 - [`dt-commons`](https://github.com/duckietown/dt-commons): likely a good choice
if the LX is pure python and doesn't require ROS.
 - [`dt-ros-commons`](https://github.com/duckietown/dt-ros-commons): likely
a good choice if you would like to use ROS in a standalone fashion (i.e., 
your exercise includes all the nodes that are needed)
 - [`dt-core`](https://github.com/duckietown/dt-core): a good choice if you 
would like to use one or more of the nodes defined in the packages 
in the `dt-core` repository. 

Coming soon:
 - ROS2 base image
 - Machine learning base image

## Duckiematrix 

By default, each LX is compatible with the [Duckiematrix](https://docs.duckietown.com/ente/devmanual-duckiematrix/intro.html).
See the [README in the LX template](https://github.com/duckietown/template-lx/blob/ente/README.md)
for more details on how to deploy this capability through the `dts code` API. 
You can define the exact configuration of the Duckiematrix environment in the 
[assets/duckiematrix](./assets/duckiematrix) directory. For more details see
[the Duckiematrix manual](https://docs.duckietown.com/ente/devmanual-duckiematrix/intro.html).

## noVNC Customization

It is also possible to customize the noVNC desktop. 
TODO: add details. 

