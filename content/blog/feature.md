+++
title = "This image contains 100% chicken"
date = 2024-09-27

[taxonomies]
tags = ["interpretability", "ml"]
+++

## Implementing feature visualization

We're going to visualize features of the ResNet18 model. During my own experimentation and testing, I obtained the best results with this model compared to InceptionV3, ResNet50 and others. Even better visualizations can be obtained with larger models such as VGG19, but at the cost of optimization speed. In this case, I'm trading off better quality for quicker feedback loops, allowing for easier experimentation.

Before implementing true hidden-layer feature visualization, we're going to do a simpler case: **class visualization**. This refers to optimizing the image for the activation of a certain output logit. Let's refer to the ResNet18's architecture again.

### Base case: Optimize the input like we're optimizing model parameters

Let's go ahead and implement the simplest, most obvious way to do class visualization. We'll refer to how we normally train neural networks. We'll often use a built-in PyTorch optimizer like Adam or Adadelta which adjusts the parameters to minimize the loss function. Here, we will try to do the same, but instead of optimizing the model's parameters, we will optimize the input.

In our case, what will the loss function be? 

Let's go ahead and try it out! We first need to download the pre-trained model:

```python
import torch
import torchvision

# This will download the model
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", weights="ResNet50_Weights.IMAGENET1K_V1")
# Set the model to evaluation mode: 
# Disables dropout and batch normalization layers, which we don't need right now
model.eval()
```

We will import matplotlib as our backend to display images conveniently. We'll also define a function `ptimg_to_mplimg` to convert the images from a PyTorch tensor to a numpy array, suitable for display in *matplotlib*. We define `show_img` to be able to display images concisely in a single function call.

```python
import matplotlib.pyplot as plt
# Converts a 
def ptimg_to_mplimg(input: torch.Tensor):
    with torch.no_grad():
        return input.detach().squeeze().permute(1, 2, 0).clamp(0, 1).numpy()
# Setting `block=False` allows us to display the progress in real time
def show_img(input: torch.Tensor, block=True):
    plt.imshow(ptimg_to_mplimg(input))
    plt.title(f'Visualization of class {target_class}')
    plt.show(block=block)
    if not block:
        plt.pause(0.001)
```

Now, with the boilerplate out of the way, we get to the meat of it.

We need to define our initial image, which will be the starting point for our optimization. We'll use uniformly-random values ranging from 0 to 1.

```python
input = torch.rand(1, 3, 299, 299, requires_grad=True)
```

We define what class we're trying to visualize. In this case, we're dealing with ResNet18, which has 1000 classes, each corresponding to one ImageNet class. This means the final layer has 1000 outputs, which  ?????????????????/. Here's [a table of all the ImageNet classes with their respective indices and labels](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/) and [a simple newline-separated list of the classes' labels](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt), in case you'd like to read the labels procedurally.

I will choose class 8: *hen*, because all of them they easily recognizable red heads. Learning rate is chosen arbitrarily, for now.

Hmm... What optimizer should we use? Let's use Adam. It's what people usually point to when training model. Why wouldn't it work now?

```python
target_class = 8	    # ImageNet class "hen"
learning_rate = 0.5
optimizer = torch.optim.Adam([input], lr=learning_rate)
```

We'll do a simple loop to perform our optimization, for (arbitrarily) 500 iterations.

```python
iterations = 500
for i in range(iterations):
    optimizer.zero_grad()
    output = model(input)
    loss = -output[:, target_class].mean()
    loss.backward()
    optimizer.step()

    # Display the image every 10 steps
    if i % 10 == 0:
        show_img(input, block=False)
        print(f"Step {i}/{iterations}")

print("Optimization ended")
show_img(input)
```

Here we have done something important: **we defined our loss function as the negative of the activation of the output neuron corresponding to the target class**. We want this neuron's activation to be as great as possible. Our optimizer tries to *minimize* the cost function, thus, if we set the cost function to be the negative of the target neuron's activation, the optimizer will try to maximize the target neuron's activation.

Okay! We've completed our first version. Let's see how it does.

<center>
  <video controls loop>
    <source src="/blog/app1.mp4" type="video/mp4">
  </video>
</center>

Hmm. That doesn't look like a fowl. Our result is extremely noisy, and it's hard to discern any particular patterns.
What can we do to improve this?

### Improvement 1: Change the optimization algorithm

We're making a mistake by using the Adam optimizer. Why? **ELABORATE**

Let's try changing the optimizer to use plain ol' SGD (stochastic gradient descent).

<center><video controls loop><source src="/blog/app2.mp4" type="video/mp4"></video></center>

That looks even worse in some way. It's noiser, and the patterns

### Improvement 2: Change the initial image

Starting with random values from 0 to 1 may be causing the optimizer to tend to extreme values (outside of the 0-1 RGB range). Not only do these values generate high contrast in the image, but if we want to get rid of the noise, starting with an already noisy image may not be the best option. Let's try a more uniformly gray initial image. To be more precise, the same random noise, but with a mean of 0.5 and a range of 0.499-0.501.

```python
input = (torch.rand(1, 3, 299, 299) - 0.5) * 0.01 + 0.5
input.requires_grad = True
```

<center><video controls loop><source src="/blog/app3.mp4" type="video/mp4"></video></center>

That's a lot better! Now if one squints, the shape of the hens can be noticed.

### Improvement 3: Enforcing transformational robustness

Every step we've been doing the backpropagation on the same image, at the same scale, rotation, and translation.
This means our code optimizes for the image to be classified as "hen" only from this perspective.
If we rotate the image, we won't be sure if the image is still classified as "hen". This means our image is not *transformationally robust*.

**Explain why this improves the noise problem**

<center><video controls loop><source src="/blog/app3.mp4" type="video/mp4"></video></center>

**Explain the code behind it**

### Improvement 4: Implement L2 regularization

<center><video controls loop><source src="/blog/app5.mp4" type="video/mp4"></video></center>

### Improvement 5: Blur the image every few steps

<center><video controls loop><source src="/blog/app6.mp4" type="video/mp4"></video></center>

## Testing our feature visualization on various classes
