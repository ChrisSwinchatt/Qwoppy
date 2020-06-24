# Qwoppy

Qwoppy is a machine learning agent that plays the game of [Qwop](http://foddy.net/Athletics.html). Qwoppy works by generating keyboard and mouse events based on the current score - as the score increases or decreases, the agent learns which input patterns are good and bad.

 * The browser is controlled using the browser automation tool [Selenium](https://www.selenium.dev/).
 * The OCR was trained using a pair of [PyTorch](https://pytorch.org/) GRU cells (encoder/decoder).
 * The agent which generates the input sequence is a PyTorch LSTM cell.
