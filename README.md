# Exercise 6: Instance Segmentation

## Setup

Run the shell script:

```
./setup.sh
```

Activate the environment:
```
conda activate 06_instance_segmentation
```

Start jupyter (you can use jupyter-notebook if you'd prefer, however jupyter-lab has some nice cell collapsing and is structured more similar to an IDE):

```
jupyter-lab
```

If running remotely (i.e through an ssh tunnel instead of no machine) call no
browser from your remote machine:

```
jupyter-lab --no-browser
```

And then set up a tunnel from your local machine (adjust port accordingly):

```
ssh -NL 8888:localhost:8888 username@hostname
```

...and continue with the instructions in the notebook.
