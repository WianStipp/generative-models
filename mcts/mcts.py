"""An implementation of Monte Carlo Tree Search."""


class MCTS:
  """
  Args:
    puct_constant: constant determining level of exploration in PUCT algorithm.
  """

  def __init__(self, puct_constant: float) -> None:
    self.puct_constant = puct_constant


class Node:

  def __init__(self) -> None:
    ...
