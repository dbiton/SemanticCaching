# adaptive_hill_climber.py

class HillClimber:
    """
    A simple 1D adaptive hill climbing optimizer.

    Adjusts a single parameter to maximize a given score, dynamically
    changing step size depending on success or failure.

    Example use:
        hc = AdaptiveHillClimber(initial_value=0.2)
        while True:
            param = hc.propose()
            score = evaluate(param)
            hc.update(score)
    """

    def __init__(
        self,
        initial_value=0.2,
        initial_step=0.1,
        min_val=0.01,
        max_val=0.5,
        min_step=0.001,
        max_step=0.5,
        step_decay=0.9,
        step_growth=1.05
    ):
        self.value = initial_value
        self.step = initial_step
        self.min_val = min_val
        self.max_val = max_val
        self.min_step = min_step
        self.max_step = max_step
        self.step_decay = step_decay
        self.step_growth = step_growth
        self.direction = 1
        self.last_score = None

    def propose(self):
        """
        Propose the next candidate value.
        """
        proposal = self.value + self.direction * self.step
        return max(self.min_val, min(self.max_val, proposal))

    def update(self, score):
        """
        Update internal state based on the new score.

        If the score improved, accept the proposal and increase the step.
        If not, reverse direction and decrease the step.
        """
        proposed = self.propose()

        if self.last_score is None or score > self.last_score:
            # Improvement
            self.value = proposed
            self.last_score = score
            self.step = min(self.step * self.step_growth, self.max_step)
        else:
            # No improvement: reverse direction and shrink step
            self.direction *= -1
            self.step = max(self.step * self.step_decay, self.min_step)
