import json

class State():
	silent = False
	transitions = []
	def __init__(self, label, emissions):
		self.label = label
		self.emissions = emissions
		if emissions == []:
			self.silent = True

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
	def __init__(self, init_state, states):
		self.init_state = init_state
		self.states = states

	def __str__(self):
		return "States: "+ "".join(str(self.states));


def load_hmm_from_json(filepath):

	with open(filepath) as data_file:
		data = json.load(data_file)
		init_state = None
		states = []
		states_dict = {}
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
			states_dict[transition.from_state].add_transition(transition)
	return HMM(init_state, states)

hmm = load_hmm_from_json("simple_hmm.json")
print(hmm)