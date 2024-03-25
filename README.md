# nth
Neural network Training Helper library for PyTorch models

## Getting Started

```shell
pip install -r requirements.txt
```

```python
python3 example.py
```

## Usage

### Import Package

```python
from nth import helper
from nth import models
```

### Define a Training Task

```python
losses = []
def train_step(batch, index, epoch):
    net.train()
    optimizer.zero_grad()
    inputs, labels = [i.to(device) for i in batch]
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

trainer = helper.Task(train_step)
trainer.run(trainloader, max_epochs=max_epochs)
```

### Register Event Handlers

#### Show Progress

```python
trainer = helper.Task(train_step)
trainer.add_event_handler(helper.Event.ITERATION_STARTED,
    lambda: print(f'Progress: {trainer.state.batch}/{len(trainloader)}', end='\r'))
trainer.run(trainloader, max_epochs=max_epochs)
```

#### Evaluate Model

```python
def evaluate_model(dataloader):
    # (skip) ...

trainer = helper.Task(train_step)
trainer.add_event_handler(helper.Event.EPOCH_COMPLETED, evaluate_model, testloader)
trainer.run(trainloader, max_epochs=max_epochs)
```
