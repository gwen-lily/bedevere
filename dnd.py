from bedevere import *
from bedevere.discrete_sums import *
from bedevere.matrix import *

advantage = 'advantage'
disadvantage = 'disadvantage'
trivantage = 'trivantage'

conditions = [advantage, disadvantage, trivantage]


class Die:

	def __init__(self, sides: int):
		self.rolls = np.arange(1, sides + 1)
		self.weights = (1/sides) * np.ones(sides)

		assert is_stochastic(self.weights)

	def roll(self, n: int = 1) -> list:
		return random.choices(self.rolls, weights=self.weights, k=n)

	def average_roll(self, n: int = 1) -> float:
		return np.average(self.rolls, weights=self.weights) * n

	def max_roll(self, n: int = 1) -> int:
		return n * max(self.rolls)


class WeightedDie(Die):

	def __init__(self, sides: int, weights: Union[np.ndarray, list]):
		super().__init__(sides)
		self.weights = weights

		assert is_stochastic(self.weights)


class GreatWeaponFightingDie(Die):

	def __init__(self, sides: int):
		super().__init__(sides)

	def roll(self, n: int = 1) -> list:
		first_rolls = super().roll(n)
		keep_rolls = list(filter(lambda x: x > 2, first_rolls))
		second_rolls = super().roll(n - len(keep_rolls))

		for roll in second_rolls:
			keep_rolls.append(roll)

		return keep_rolls

	def average_roll(self, n: int = 1) -> float:
		N = self.max_roll()
		mean_weights = list(map(lambda x: 2/N**2 + (1/N)*(x > 2), self.rolls))
		assert is_stochastic(np.asarray(mean_weights))

		return np.average(self.rolls, weights=mean_weights) * n


D4 = Die(4)
D6 = Die(6)
D8 = Die(8)
D10 = Die(10)
D12 = Die(12)
D20 = Die(20)


def generic_roll(die: Die, n: int = 1, bonus: int = 0) -> int:
	return sum(die.roll(n)) + bonus


def roll_d20(bonus: int = None, condition: str = None) -> int:
	die = D20

	if condition == advantage:
		roll = max(die.roll(2))

	elif condition == disadvantage:
		roll = min(die.roll(2))

	elif condition == trivantage:
		roll = max(die.roll(3))

	else:
		roll = die.roll()[0]

	if roll == 1:
		raise CriticalFumble(roll, roll + bonus)

	elif roll == 20:
		raise CriticalHit(roll, roll + bonus)

	else:
		return roll + bonus
