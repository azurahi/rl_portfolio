from random import random
import numpy as np
import math
import pandas as pd

import gym
from gym import spaces

import hyperparameter

class MarketEnv(gym.Env):
	PENALTY = 1  # 0.999756079

	def __init__(self, dir_path, target_codes, input_codes, start_date, end_date, scope=60, sudden_death=-1.,
				 cumulative_reward=False):
		self.startDate = start_date
		self.endDate = end_date
		self.scope = scope
		self.sudden_death = sudden_death
		self.cumulative_reward = cumulative_reward

		self.inputCodes = []
		self.targetCodes = []
		self.dataMap = {}
		print(target_codes)
		print(input_codes)
		for code in list(set(target_codes + input_codes)):
			fn = dir_path + code + ".csv"
			data = {}
			lastClose = 0
			lastVolume = 0
			try:
				f = open(fn, "r",encoding= 'euc-kr')
				next(f)
				for line in f:
					if line.strip() != "":
						dt, openPrice, close, volume, per, volume_ind, volume_comp, volume_fore = line.strip().split(",")
						try:
							if dt >= start_date:
								# high = float(high) if high != "" else float(close)
								# low = float(low) if low != "" else float(close)
								openPrice = float(openPrice)
								close = float(close)
								per = 0.0 if per == "" else float(per)
								volume_ind = int(volume_ind)
								volume_comp = int(volume_comp)
								volume_fore = int(volume_fore)
								volume = float(volume)

								if lastClose > 0 and close > 0 and lastVolume > 0:
									openPrice_ = (openPrice -lastClose) / lastClose
									close_ = (close - lastClose) / lastClose
									# high_ = (high - close) / close
									# low_ = (low - close) / close
									volume_ = (volume - lastVolume) / lastVolume
									per_ = per
									volume_ind_ = (volume_ind - volume)/volume
									volume_comp_ = (volume_comp - volume)/volume
									volume_fore = (volume_fore -volume)/volume

									data[dt] = (openPrice_, close_, volume_,per_,volume_ind_,volume_comp_, volume_fore)

								lastClose = close
								lastVolume = volume
						except Exception as e:
							print(e, line.strip().split(","))
				f.close()
			except Exception as e:
				print(e)

			if len(data.keys()) > scope:
				self.dataMap[code] = data
				data_d = pd.DataFrame(data)
				data_d.to_csv('test.csv')
				if code in target_codes:
					self.targetCodes.append(code)
				if code in input_codes:
					self.inputCodes.append(code)

#descrete한 Action Space에 대하여 연속적인 행동을 취할 수 있도록 변화시키도록 코드를 변환한다.


		self.actions = [
			"LONG",
			"SHORT",
		]

		self.action_space = spaces.Discrete(len(self.actions))
		self.observation_space = spaces.Box(np.ones(scope * (len(input_codes) + 1)) * -1,
											np.ones(scope * (len(input_codes) + 1)))

		self.reset()
		self._seed()

	def _step(self, action):
		if self.done:
			return self.state, self.reward, self.done, {}

		self.reward = 0
		if self.actions[action] == "LONG":
			if sum(self.boughts) < 0:
				for b in self.boughts:
					self.reward += -(b + 1)
				if self.cumulative_reward:
					self.reward = self.reward / max(1, len(self.boughts))

				if self.sudden_death * len(self.boughts) > self.reward:
					self.done = True

				self.boughts = []

			self.boughts.append(1.0)
		elif self.actions[action] == "SHORT":
			if sum(self.boughts) > 0:
				for b in self.boughts:
					self.reward += b - 1
				if self.cumulative_reward:
					self.reward = self.reward / max(1, len(self.boughts))

				if self.sudden_death * len(self.boughts) > self.reward:
					self.done = True

				self.boughts = []

			self.boughts.append(-1.0)
		else:
			pass

		vari = self.target[self.targetDates[self.currentTargetIndex]][0]
		self.cum = self.cum * (1 + vari)

		for i in range(len(self.boughts)):
			self.boughts[i] = self.boughts[i] * MarketEnv.PENALTY * (1 + vari * (-1 if sum(self.boughts) < 0 else 1))

		self.defineState()
		self.currentTargetIndex += 1
		if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[
			self.currentTargetIndex]:
			self.done = True

		if self.done:
			for b in self.boughts:
				self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1
			if self.cumulative_reward:
				self.reward = self.reward / max(1, len(self.boughts))

			self.boughts = []

		return self.state, self.reward, self.done, {"dt": self.targetDates[self.currentTargetIndex], "cum": self.cum,
													"code": self.targetCode}

	def _reset(self):
		print(self.targetCodes)
		self.targetCode = self.targetCodes[int(random() * len(self.targetCodes))]
		self.target = self.dataMap[self.targetCode]
		self.targetDates = sorted(self.target.keys())
		self.currentTargetIndex = self.scope
		self.boughts = []
		self.cum = 1.

		self.done = False
		self.reward = 0

		self.defineState()

		return self.state

	def _render(self, mode='human', close=False):
		if close:
			return
		return self.state

	'''
	def _close(self):
		pass

	def _configure(self):
		pass
	'''

	def _seed(self):
		return int(random() * 100)

	def defineState(self):
		tmpState = []

		budget = (sum(self.boughts) / len(self.boughts)) if len(self.boughts) > 0 else 1.
		size = math.log(max(1., len(self.boughts)), 100)
		position = 1. if sum(self.boughts) > 0 else 0.
		tmpState.append([[budget, size, position]])

		'''
		Observation을 관리하는 칸 
		여기서 포트폴리오 입력과 State Data를 정리하는 것
		'''


		subject = []
		subject1 = []
		subject2 = []
		subject3 = []
		subject4 = []
		subject5 = []
		subjectVolume = []
		for i in range(self.scope):
			try:
				# print(self.target)
				# subject.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][3]])
				subject1.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][5]])
				subject2.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][2]])
				subject3.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][0]])
				# subject4.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][5]])
				subject4.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][6]])
				# subject5.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][0]])
				subjectVolume.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][4]])
			except Exception as e:
				print(self.targetCode, self.currentTargetIndex, i, len(self.targetDates))
				self.done = True
		tmpState.append([[subject1,subject2, subject3,subject4, subjectVolume]])

		tmpState = [np.array(i) for i in tmpState]

		self.state = tmpState
