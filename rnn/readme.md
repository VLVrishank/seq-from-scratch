# Recurrent Neural Network (RNN)

A simple neural network that can process sequences by remembering what it saw before.

## What's the Problem?

Regular neural networks can't handle sequences well. If you feed them a sentence word by word, they forget the previous words by the time they see the next one. It's like reading a book but forgetting every page as soon as you turn it.

## How RNNs Work

RNNs solve this by having **memory**. Each time they process a new input, they:

1. Look at the current input
2. Remember what they learned from previous inputs (hidden state)
3. Combine both to make a decision
4. Update their memory for the next input

Think of it like taking notes while reading - you keep track of what happened before to understand what's happening now.

## The Math (No Jargon Version)

For each word in a sentence:
new_memory = tanh(input * W1 + old_memory * W2 + bias)
output = new_memory * W3 + bias

That's it! The network learns the best values for W1, W2, W3 during training.
