from bedevere import *
from bedevere.discrete_sums import *
from bedevere.matrix import *
import re

advantage = 'advantage'
disadvantage = 'disadvantage'
trivantage = 'trivantage'

conditions = [advantage, disadvantage, trivantage]
roll_generic_pattern = re.compile(r'(\d*)d(4|6|8|10|12)', re.I)
whitespace_pattern = re.compile(r'\s')


def dice_string_parser(dice_string: str):
	"""Parse a dice string into one or more tuples of information for die and dice construction

	Separate types of dice separated by any number of spaces and a + or -."""

	dice_type_tuples = whitespace_pattern.sub('', dice_string.strip()).split('+')

	dice_list = []

	for dtt in dice_type_tuples:
		n, sides = roll_generic_pattern.groups()
		n = int(n) if n else 1
		sides = int(sides)

		for i in range(n):
			dice_list.append(Die(sides))

	dice = Dice(dice_list)


class Die:

	def __init__(self, sides: int):
		self.rolls = np.arange(1, sides + 1)
		self.weights = (1/sides) * np.ones(sides)

		assert is_stochastic(self.weights)

	def roll(self, n: int = 1) -> list:
		return random.choices(self.rolls, weights=self.weights, k=n)

	def average_roll(self, n: int = 1) -> float:
		return np.average(self.rolls, weights=self.weights) * n

	def min_roll(self, n: int = 1) -> int:
		return n * min(self.rolls)

	def max_roll(self, n: int = 1) -> int:
		return n * max(self.rolls)

	def to_dict(self) -> dict:
		d = {}

		for roll, weight in zip(self.rolls, self.weights):
			d[roll] = weight

		return d


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


class Dice(Die):

	def __init__(self, dice: Union[Die, List[Die]]):
		super().__init__(1)

		if isinstance(dice, Die):
			self.dice = [dice]

		else:
			self.dice = dice

		minimum_roll = sum(map(lambda die: die.min_roll(), self.dice))
		maximum_roll = sum(map(lambda die: die.max_roll(), self.dice))
		list_of_distributions = list(map(lambda die: die.to_dict(), self.dice))
		roll_distribution = convolution_list(list_of_distributions)

		assert minimum_roll == min(roll_distribution.keys())
		assert maximum_roll == max(roll_distribution.keys())

		self.rolls = np.arange(minimum_roll, maximum_roll + 1)
		self.weights = np.asarray(list(map(lambda r: roll_distribution[r], self.rolls)))

		assert is_stochastic(self.weights)

	def roll(self, n: int = 1) -> list:
		unflattened_rolls = list(map(lambda d: d.roll(n), self.dice))
		return [r for roll in unflattened_rolls for r in roll]


class Weapon:

	def __init__(self, name: str, attack_bonus: int, damage_bonus: int, damage_dice: Die):
		self.name = name
		self.attack_bonus = attack_bonus
		self.damage_bonus = damage_bonus
		self.damage_dice = damage_dice

	def attack(self, condition: str = None, extra_damage_dice: Die = None, extra_damage_bonus: int = 0) \
			-> Tuple[int, int, object]:

		main_damage = 0
		extra_damage = 0
		is_Critical_Hit = False
		C = None

		try:
			roll_to_hit = roll_d20(self.attack_bonus, condition)

		except CriticalHit as C:
			is_Critical_Hit = True

		except CriticalFumble as C:
			pass

		main_damage = roll_dice(self.damage_dice, self.damage_bonus, is_Critical_Hit)
		extra_damage = roll_dice(extra_damage_dice, extra_damage_bonus, is_Critical_Hit)
		return main_damage, extra_damage, C


D4 = Die(4)
D6 = Die(6)
D8 = Die(8)
D10 = Die(10)
D12 = Die(12)
D20 = Die(20)
D100 = Die(100)


def roll_generic(die: Die, n: int = 1, bonus: int = 0) -> int:
	return sum(die.roll(n)) + bonus


def roll_dice(dice: Die, bonus: int = 0, is_critical: bool = False) -> int:
	dice_multiplier = 2 if is_critical else 1
	return sum(dice.roll(dice_multiplier)) + bonus


def roll_d20(bonus: int = None, condition: str = None) -> int:

	if condition == advantage:
		roll = max(D20.roll(2))

	elif condition == disadvantage:
		roll = min(D20.roll(2))

	elif condition == trivantage:
		roll = max(D20.roll(3))

	else:
		roll = D20.roll()[0]

	if roll == 1:
		raise CriticalFumble(roll, roll + bonus)

	elif roll == 20:
		raise CriticalHit(roll, roll + bonus)

	else:
		return roll + bonus

# Legacy support
# Support until 1.0.0


def generic_roll(die: Die, n: int = 1, bonus: int = 0) -> int:
	warnings.warn("Method will change to roll_generic in 1.0.0", FutureWarning)
	return roll_generic(die, n, bonus)


def test():
	weapon = Weapon(5, 4, [D6, D6])

	for condition in conditions:
		main_damage, extra_damage, Crit = weapon.attack(condition, extra_damage_dice=Dice([D6, D8, D8, D8]))

		if type(Crit) == CriticalHit:
			print('crit', main_damage, extra_damage)

		elif type(Crit) == CriticalFumble:
			print('"fumble" - John Madden', main_damage, extra_damage)


if __name__ == '__main__':
	test()
