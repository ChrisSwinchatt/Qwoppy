# Qwoppy

Qwoppy is a machine learning agent that plays the game of [Qwop](http://foddy.net/Athletics.html). Qwoppy works by generating keyboard and mouse events based on the current score - as the score increases or decreases, the agent learns which input patterns are good and bad.

 * The browser is controlled using the browser automation tool [Selenium](https://www.selenium.dev/).
 * The OCR was trained using a pair of [PyTorch](https://pytorch.org/) GRU cells (encoder/decoder).
 * The agent which generates the input sequence is a PyTorch LSTM cell.

## OCR

### Hyperparameters

 * Learn metres: true or false
 * Scan image in column major (left-to-right) or row major (top-to-bottom) order
 * Number of epochs: 200
 * Batch size: 19 batches of 83 samples
 * Encoder: 1x256 unit GRU, Adam optimizer with lr=0.0001
 * Decoder: 1x256 unit GRU, Adam optimizer with lr=0.0001
 * Hidden state: random
 * Loss function: MSE

#### Comparison

Ordering     | With metres | Loss@E50 | Loss@E100 | Loss@E200  | Loss@Test | Acc@Test
-------------|-------------|----------|-----------|------------|-----------|---------
Column major | Yes         | 0.00998  | 0.00738   | 0.00676    | 0.00658   | 87%
Column major | No          | 0.00706  | 0.00673   | 0.00652    | 0.00641   | 87%
Row major    | Yes         | 0.00969  | 0.00710   | 0.00678    | 0.0066    | 87%
Row major    | No          | 0.00633  | 0.00495   | 0.00097    | 0.00126   | 98%
