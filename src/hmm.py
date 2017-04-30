import json
import hmm_util

class State():
	silent = False
	transitions = []
	def __init__(self, label, emissions):
		self.label = label
		self.emissions = emissions
		if emissions == []:
			self.silent = True

	def get_emission_prob(self, observation):
		r = 0
		for emission in self.emissions:
			if emission[1] == observation:
				r = emission[0]
		return r

	def add_transition(self,transition):
		self.transitions += [transition]

	def __repr__(self):
		return self.label + "\n"

	def __str__(self):
		return self.label
		#return ("State: " + self.label + 
				#"\n\tTransitions\n:" + "\n\t\t".join(self.transitions) + 
				#"\tEmissions: \n" + "\n\t\t".join(self.emissions))


class Transition():
	def __init__(self, from_state, to_state, probability):
		self.probability = probability
		self.from_state = from_state
		self.to_state = to_state

class HMM():
	states = []
	transitions = []
	def __init__(self, init_state, states, transitions):
		self.init_state = init_state
		self.states = states
		self.transitions = transitions

	def __str__(self):
		return "States: "+ "".join(str(self.states));

def forward(hmm, sequence):
	i = 0
	state_mapping = {}
	inverse_mapping = [None for _ in range(len(hmm.states))]
	for state in hmm.states:
		state_mapping[state.label] = i
		inverse_mapping[i] = state
		i += 1
	transition_matrix = [[0 for _ in range(len(hmm.states))] for _ in range(len(hmm.states))]
	for transition in hmm.transitions:
		from_index = state_mapping[transition.from_state]
		to_index = state_mapping[transition.to_state]
		transition_matrix[from_index][to_index] = transition.probability
	matrix = []
	init_index = state_mapping[hmm.init_state.label]
	print(init_index)
	print(state_mapping)
	print(transition_matrix)
	print(inverse_mapping)
	forward_matrix = [[0 for _ in range(len(sequence))] for _ in range(len(hmm.states))]

	for i in range(len(hmm.states)):
		forward_matrix[i][0] = 0
	forward_matrix[init_index][0] = 1
	for t in range(1,len(sequence)):
		for j in range(len(hmm.states)):
			sum = 0
			for i in range(len(hmm.states)):
				sum += forward_matrix[i][t-1] * transition_matrix[i][j]
			forward_matrix[j][t] = sum * inverse_mapping[j].get_emission_prob(sequence[t])

	sum = 0
	for j in range(len(hmm.states)):
		sum += forward_matrix[j][-1]
	print(sum)
	print(forward_matrix)
	return sum

def load_hmm_from_json(filepath):

	with open(filepath) as data_file:
		data = json.load(data_file)
		init_state = None
		states = []
		states_dict = {}
		transitions = []
		for state_data in data["states"]:
			label = state_data["label"]
			emission_data = state_data["emissions"]
			state = State(label,emission_data)
			states += [state]
			if label == data["init_state"]:
				init_state = state
			states_dict[label] = state
		for t_data in data["transitions"]:
			transition = Transition(t_data[0], t_data[1], t_data[2])
			transitions += [transition]
			states_dict[transition.from_state].add_transition(transition)
	return HMM(init_state, states, transitions)

hmm = load_hmm_from_json("simple_hmm.json")
print(hmm)
forward(hmm, [1,3,3])