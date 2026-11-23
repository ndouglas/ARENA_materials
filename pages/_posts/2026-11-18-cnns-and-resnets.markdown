---
layout: post
title:  "CNNs & ResNets (0.2)"
date:   2026-11-18 06:04:18 -0500
categories: arena
---
Moving on!

In 0.2, we implement a few ML primitives and composites, namely [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit), an [affine transformation](https://en.wikipedia.org/wiki/Affine_transformation), a simple multilayer perceptron, etc.

## Things to Look Up, Read, Watch, Etc
- uniform Kaiming initialization
- Xavier initialization
- [Intuitively Understanding Cross Entropy](https://www.youtube.com/watch?v=Pwgpl9mKars)
- [Intuitively Understanding KL Divergence](https://www.youtube.com/watch?v=SxGYPqCgJWM)
- [But What Is a Convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)
- [A Comprehensive Guide to Convolutional Neural Networks - the ELI5 Way](https://medium.com/data-science/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
- [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/)
- [Understanding 2D Convolutions in PyTorch](https://medium.com/@ml_dl_explained/understanding-2d-convolutions-in-pytorch-b35841149f5f)
- [Batch Normalization in Convolutional Neural Networks](https://www.baeldung.com/cs/batch-normalization-cnn)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)

## Funny Little Device Gotcha

I fumbled my way through implementing ReLU - I understood in principle what to do, but not the best way to accomplish it. I ended up with this:

```python
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.max(x, t.zeros(*x.shape))
```

which worked, and passed the tests, but then I encountered this during training:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
```

I figured out what had happened, and scrolled back up the page to change it to the official solution:

```python
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.max(x, t.tensor(0.0))
```

But it just goes to show you 😩 - you do an innocuous thing, trying to be somewhat smart, and it bites you in the 🍑 an hour later.

After I fixed that:

![Graph of cross-entry loss over samples seen while learning from the MNIST data set](/assets/images/2026-11-18-01-training.png)

## No, I Didn't Know We'd Gotten That Far Yet

Another boneheaded thing I did was to lose the plot entirely during the validation loop exercise.

I knew _in theory_ how the prediction works - you take an argmax over the logits and that indicates the predicted digit. For no good reason, when implementing the validation loop, I measured loss instead, and was nonplussed when I saw the following graph:

![Is this loss?](/assets/images/2026-11-18-02-is-this-loss.png)

I blame it on the simplicity of the interface. It's still continually shocking to me how _simple_ the broad strokes of machine learning are. There's obviously a lot I still don't grok (like, you know, anything beyond the broad strokes), but it's all shockingly elegant.

When I took an AI class in college - and was one of very few students in the class, mind you - neural networks seemed a bit ephemeral. Part of that was that my professor downplayed them and seemed irritated even to have to bring them up. But then, he was a deeply unpleasant man and that impacted _everything_ he taught. For all I know, he's at a frontier lab now.

Anyway, I felt much better once I saw this:

![No, this is.](/assets/images/2026-11-18-03-no-this-is.png)
