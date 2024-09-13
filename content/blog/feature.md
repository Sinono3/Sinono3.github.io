+++
title = "A Brief, Practical Introduction to Feature Visualization"
date = 2024-09-27

[taxonomies]
tags = ["interpretability", "ml"]
+++

Growing more capable by the second, AI is being adopted by both industries and governments as a mainstream technology.
AI allows automation of knowledge work, and with enough data, compute and algorithmic improvements, AI can replace engineering,
research, and administrative jobs.

This could be great or detrimental, depending on how effectively these technologies achieve our goals.
There's a bunch of reasons as to why this is, among them: 

1. We don't understand how deep learning models *actually* work.
2. We don't know how to make deep learning models do *precisely* what we want.
3. We don't even know what we want. (Or rather, how to explicitly state what we want)
4. We don't know which regulations reduce AI misuse most effectively.
5. *...many more*

However, explaining these problems is not why I'm writing this post -- and honestly, I have no idea how to...
Except for problem 1: interpretability, which I do know a bit about.

<!-- TODO: You can check out these resources to answer the other questions... -->

> **Heads-up**: I assume some understanding of [feed-forward neural networks (MLP)](https://www.youtube.com/watch?v=aircAruvnKk), [gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) and [what *softmax* is](https://youtu.be/wjZofJX0v4M?si=w6FBaLX8KNRzk_If&t=1342).

## What does interpretability solve?

Let's go back to the problem 1:

> We don't understand how deep learning models *actually* work.

What do I mean by "how they *actually* work"?

For decades, we have built different deep learning architectures such as multi-layer perceptrons; convolutional, residual and recurrent neural networks; LSTMs; transformers; and many more. During training, these machines encode patterns and algorithms inside their parameters (weights, biases, etc). We know how they do this, but not precisely what these patterns and algorithms are.

Once trained, **models are black boxes**: while we provide inputs and get mostly correct outputs, we don't understand how the model did so at a neuron level. We know that neurons are connected, but *how* are they connected to achieve the goal?

I'll present you with an example of what I mean. Consider [this cat](https://unsplash.com/photos/black-and-white-cat-lying-on-brown-bamboo-chair-inside-room-gKXKBY-C-Dk).

{{ img(path="@/blog/cat.jpg", caption="[*(Consider them.)*](https://aisafety.dance/)") }} 

If fed to an image classification model such as ResNet18, it would be classified as "Egyptian cat," which wouldn't be far from the truth. There's no single "cat" category, so it would be impossible for the model to simply answer "cat". So, practically, it's correct!

**FEED CAT TO RESNET18. SHOW LOGITS, SOFTMAX, ARGMAX AND THEN LOOKUP IN LIST OF IMAGENET CLASSES**

But how did the model come to that conclusion? Let's do a reverse analysis.

- Since the ImageNet class "Egyptian cat" has an index of 285, we know that in the fully connected last layer of the model (`fc`), neuron 285 is the greatest among the neurons in that layer (which are 1000 in total).
- Neuron 285 in `fc` was activated because of some neuron activations in the previous layer (`avgpool`).
- Neurons in `avgpool` that contributed to neuron 285 in `fc` come from the results of a convolutional layer. 
- This convolutional layer outputted its result from another convolutional layer... 
- And so on... until we get to the first layer, which is connected directly to the input image (cute cat pic).

From this analysis, multiple questions pop up:

1. **Circuit identification:** Which neurons in `avgpool`, when activated, cause neuron 285 of layer `fc` to activate? And in the previous layer? What complex circuit of neurons has formed across the network's layers to conclude this image corresponds to an "Egyptian cat"?
2. **Visualization:** What do these neurons firing represent? Do they correspond to concepts, shapes, forms, or objects? Can we see an image of what a single neuron represents? What about a set of neurons?
3. **Attribution**: What parts of the input image contributed to the model outputting "Egyptian cat"? Which ones didn't? What parts of the input image contribute to a particular neuron firing? What images make a neuron fire?

At the surface level, interpretability tries to answer these kinds of questions.

Today, we're going to have our try at **visualization**.

## Defining visualization

In the general case of any network, we can define *feature visualization* as generating an input that maximizes the activation of a part of the network: an output neuron, a hidden-layer neuron, a set of neurons, or an entire layer.

In the case of image classification models, feature *visualization* refers to generating an image that maximizes the activation of a part of a network. Let's say we do *class* visualization, where we optimize an image so the model *overwhelmingly* classifies it in a particular class (meaning the neuron corresponding to that class in the last layer will be significantly activated, more than all the other output neurons.) 

In a perfect world, if we were to visualize class 285 on ResNet18, we would get an image of a cute kitten. In reality, though, feature visualizations can be confusing and unintelligible compared to a natural picture. We'll see this as we try to implement it ourselves.

## Implementing visualization

Using PyTorch, let's try to implement class visualization for a pre-trained image classification model. We're going to choose a specific ImageNet class and optimize an image so the model classifies it in the specified class. So, which class are we choosing?

{{ img(path="@/blog/hen.jpg", caption="ImageNet class 8: *hen*. [Source.](https://unsplash.com/photos/brown-and-red-he-n-G61iAuzI9NQ)") }} 

Why chickens? Because **all** of them they easily recognizable red combs. Thus, it will be easier to see if our visualization works at all from the get-go.

> **In case you want to use another ImageNet class**, [here's the list you can choose from](https://github.com/pytorch/hub/blob/c7895df70c7767403e36f82786d6b611b7984557/imagenet_classes.txt). Once you did, record the line number of the label and subtract 1 to get the output neuron or class index. (This is because line numbers start at 1, while PyTorch tensors indexes do at 0)

We're going to visualize the ResNet18 model. I obtained "good" results with this model during my own experimentation. Better visualizations can be obtained with larger models such as VGG19, but at the cost of optimization speed. In this case, I'm trading off better quality for quicker feedback loops, allowing easier experimentation.

### Base case: Optimize the input like we're optimizing model parameters

We will begin writing code by importing matplotlib as our backend to display images conveniently. We'll also define a function `ptimg_to_mplimg` to convert the images from a PyTorch tensor to a numpy array, suitable for display in matplotlib. We define `show_img` to be able to display images concisely in a single function call.

```python
import matplotlib.pyplot as plt
def ptimg_to_mplimg(input: torch.Tensor):
    with torch.no_grad():
        return input.detach().squeeze().permute(1, 2, 0).clamp(0, 1).numpy()
# Setting `block=False` allows us to display the progress in real-time
def show_img(input: torch.Tensor, block=True):
    plt.imshow(ptimg_to_mplimg(input))
    plt.title(f'Visualization of class {target_class}')
    plt.show(block=block)
    if not block:
        plt.pause(0.001)
```

With the boilerplate out of the way, let's go ahead and implement the simplest, most obvious way to do class visualization. We'll refer to how we normally train neural networks: use a built-in PyTorch optimizer which adjusts parameters to minimize the loss function. Here, we will try to do the same, but instead of optimizing the model's parameters, we will optimize the input. "What will be our loss function?" you may ask. We'll answer that later.

Let's download the pre-trained model:

```python
import torch
import torchvision

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", weights="ResNet18_Weights.IMAGENET1K_V1")
# Set the model to evaluation mode: 
# Disables dropout and batch normalization layers, which we don't need right now
model.eval()
```

We need to define our initial image, which will be the starting point for the optimization. We'll use uniformly random values ranging from 0 to 1. Do notice that we can use any image as the starting point.

```python
input = torch.rand(1, 3, 299, 299, requires_grad=True)
```

We declare our target class to be "hen".

```python
target_class = 8 # ImageNet class "hen"
```

Hmm, what optimizer should we use? Why not plain ol' SGD? (stochastic gradient descent)

```python
learning_rate = 0.5
optimizer = torch.optim.SGD([input], lr=learning_rate)
```

We'll create a function that performs a single optimization step to organize our code neatly.

```python
def step(model, optimizer: torch.optim.Optimizer, input: torch.Tensor):
    optimizer.zero_grad()
    output = model(input)
    loss = -output[:, target_class].mean()
    loss.backward()
    optimizer.step()
    return input
```

Here, we have done something important: **We defined our loss function as the negative of the activation of the output neuron corresponding to the target class**. We want this neuron's activation to be as great as possible. Our optimizer tries to *minimize* the cost function, thus, if we set the cost function to be the negative of the target neuron's activation, the optimizer will try to maximize the target neuron's activation.

Okay! We've completed our first version. Let's see how it does.

<center><video controls loop><source src="/blog/app0.mp4" type="video/mp4"></video></center>

Hmm. That doesn't quite look like a fowl. What can we do to improve this?

### Improvement 1: Change the initial image

Starting with random values from 0 to 1 may cause the optimizer to tend to extreme values (outside the 0-1 RGB range). Not only do these values generate high contrast in the image, but if we want to get rid of the noise, starting with an already noisy image may not be the best option. Let's try a more uniformly gray initial image. To be more precise, the same random noise, but with a mean of 0.5 and a range of 0.499-0.501.

```python
input = (torch.rand(1, 3, 299, 299) - 0.5) * 0.01 + 0.5
input.requires_grad = True
```

<center><video controls loop><source src="/blog/app1.mp4" type="video/mp4"></video></center>

That's a lot better! If one squints, the red combs of the chickens pop out, while in the rest of the image, feather-like patterns start to emerge.

### Improvement 2: Enforcing transformational robustness

We've been doing backpropagation on the same image at the same scale, rotation, and translation for every step.
This means our code optimizes for the image to be classified as "hen" only from this perspective.
Nothing assures us that if we rotate the image, the image will still be classified as "hen." 
This means our image is not *transformationally robust*.

**Explain why this improves the noise problem**

<center><video controls loop><source src="/blog/app2.mp4" type="video/mp4"></video></center>

**Explain the code behind it**

### Improvement 3: Implement L2 regularization

<center><video controls loop><source src="/blog/app3.mp4" type="video/mp4"></video></center>

### Improvement 4: Blur the image every few steps

<center><video controls loop><source src="/blog/app4.mp4" type="video/mp4"></video></center>

## Testing our feature visualization on various classes


## Real-world case studies

## Limitations


## Conclusion

Okay, that's good enough for the scope of this post. I hope you found interpretability to be fun and interesting. 

But hey, if you did, don't stop here! I barely scratched the surface of what the field actually revolves around. There are a bunch of resources for you to continue researching.

- [Distill's Circuits Thread](https://distill.pub/2020/circuits/)
- [Transformer Circuits Thread](https://transformer-circuits.pub/)

