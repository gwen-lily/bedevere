from bedevere import *


class Critical(Exception):

	def __init__(self, rolls: list, value: int, message: str = 'You rolled a critical value.'):
		self.rolls = rolls
		self.value = value
		self.message = message
		super().__init__(self.message)


class CriticalHit(Critical):

	def __init__(self, rolls: list, value: int, message: str = 'You scored a critical hit!'):
		super().__init__(rolls, value, message)


class CriticalFumble(Critical):

	def __init__(self, rolls: list, value: int, message: str = 'You scored a critical fumble!'):
		super().__init__(rolls, value, message)
