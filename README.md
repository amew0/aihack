
## Convolutional Layers

```python
>>> nn.Conv2d (i, o , k , s , p): # INPUT (N x i x s x s) --> OUTPUT (N x o x s' x s')
```
    i: input
    o: output
    k: kernel
    s: stride
    p: padding

    s' = int((s - k + 2*p) / s + 1)   # round down (floor)

    Always, check with Random Values to make sure the Conv2d is well setup.

    Example:
    >>> layer1 = nn.Conv2d (i=1, o=10 , k=3 , s=2 , p=1)

    >>> layer1(torch.rand([32,1,512,512])) 
    # Interpretation: If my input is 32 images/batch of 1 channel of dimension 512 x 512,
    and I pass them through layer1, what do I get?
    # You get: 32 x 10 x 256 x 256: 
        what is this?
            For each 32 images, you get 10 features extracted and each of them is of dimension 256 x 256

    * experiment with k, s, p

## Model

    It usually extends nn.Module # So we don't need to build the model scratch Only two functions should be replaced:
        >>> def __init__ ()
            # model architecture is here
        >>> def forward (self,x)
            # glueing the architecture and returning expected output

## Criterion

    >>> criterion = nn.CrossEntropyLoss() # EXPLORE other Loss metric also
    >>> optimizer = optim.SGD(model.parameters(), lr=0.01) # EXPLORE OTHER optimizers also

    Initial parameters of my model are Random, so at first it will guess the output and will get it right at a probability of (1/no.of classes)
    So, what do I need to do?

        >>> loss = criterion(outputs, labels)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

        1. Calculate how far off it is. (each predicted with each label and count how many of them did it get them right) 
            >>> criterion(outputs, labels)
        2. Change the values a little bit (Learning Rate). 
            >>> loss.backward()

        
## Model Validation (and Evaluation):

    Pytorch by default is designed for training. So we have to explicitly stop training by,
    with torch.no_grad(): # What happens if we don't? Gradients / changes are ready to be appended to the parameters. (and we don't want that in this stage)

## Transfer Learning

    Why do we need to bother tweaking the weights when it was already done by others partially?

    VGG16, RESNET[18/50/etc.], and others Popular ones for our TASK
    But we have to adjust them a little to make them compatible, like
        - Input dimenstion expectation
        - Output dimension requirement
    But, takes more time to train 


## Saving

    For identification save the filename with the team name and anything after that .pt. e.g. Z_idontknowwhattowritehere.pt

    !!IMPORTANT!! To make it compatible with the testing experiment save it ONLY THIS WAY,
    
    x = torch.Tensor(1, 3, 300, 400) # Batch=1 Color-Channel=3 Dimenstion-width=300 Height=400 (update this as required and used by the model)
    with torch.no_grad():
        traced_cell = torch.jit.trace(model, (x))
    model_name = "./models/Y_model_1.pt"
    torch.jit.save(traced_cell, model_name)

## MOMENT OF TRUTH

    Upload it and see how you performed in the testing dataset. 
    Good Luck
