# draft_assistant
Premier draft assistant for MTGArena

# Installation
Once you clone or download the repository, `cd` inside it and
- ideally start a virtual environment with `python -m venv venv`
- activate it (on Mac) `source venv/bin/activate`
- install dependencies with `pip install -r requirements.txt`
- run it with `python main.py`. Note: if you start the application before starting Arena, it will process the currently available log file until it is overwritten with the next one.

# What it does
Compared to other 17Lands-based draft Assistants, we are not using heruristics like GameInHand WinRate etc. We are using pure data-based approaches to model the probability distribution `P(x|y,z,w,k)`, i.e. the probability of picking card `x` from pack `y`, conditioned on pack `y`, pool of cards `z`, expected number of wins `w` and player rank `k`.
At training time, we model such probability function with a modified "cross" attention mechanism between the cards in the pool and the cards in the pack, taken from the transformer architecture of Large Language Models (although we keep the model extremely tiny at about 24KB - and trained on a Macbook Air!). All the variables in the conditioning of the probability can be extracted from the Draft Data on [17Lands]([url](https://www.17lands.com/public_datasets)).
At inference time, we force the expected wins to 7 and the expected rank to be mythic, to condition on the best possible choice. This is strictly better than training a model to mimic the average mythic player.
