from pomegranate import *

Graduate = DiscreteDistribution({'graduate':0.9, 'no-graduate':0.1})
Offer = ConditionalProbabilityTable([
   ['graduate', 'o1', 0.5],
   ['graduate', 'o2', 0.75],
   ['no-graduate', 'o1', 0.05],
   ['no-graduate', 'o2', 0.25]], [Graduate])
   
s_graduate = State(Graduate, 'graduation status')
s_offer_1 = State(Offer, 'offer1')
s_offer_2 = State(Offer, 'offer2')
model = BayesianNetwork('graduation status')
model.add_states(s_graduate, s_offer_1, s_offer_2)
model.add_transition(s_graduate, s_offer_1)
model.add_transition(s_graduate, s_offer_2)
model.bake() # finalize the topology of the model

#print(model)
#print()

print ('The number of nodes:', model.node_count())
print ('The number of edges:', model.edge_count())

# predict_proba(Given factors)

print()
# part a - P(o2|g, ~o1)
print("a) P(o2 | g, ~o1)")
print (model.predict_proba({'graduate':'offer1'})[1].parameters)
print()

# part b - P(g|o1, o2)
print("b) P(g | o1, o2)")
print (model.predict_proba({'offer1':'o1','offer2':'o2'})[0].parameters)
print()

# part c
print("c) P(g | ~o1, o2)")
print (model.predict_proba({'offer1':'o2','offer2':'o2'})[0].parameters)
print()

# part d
print("d) P(g | ~o1, ~o2)")
print (model.predict_proba({'offer1':'o2','offer2':'o2'})[0].parameters)
print()

# part e
print("e) P(o2 | o1)")
print (model.predict_proba({'graduation status':'o1'})[1].parameters)
print (model.predict_proba({'no-graduate':'o1'})[1].parameters)
print()


Happiness = DiscreteDistribution({'sunny':0.7, 'raise':0.01})
day = ConditionalProbabilityTable([
   ['happiness', 'raise', 0.0184],
   ['happiness', 'sunny', 0.9],
   ['sunny', 'raise', 0.01],
   ['happiness', 'notsunny', 0.3]], [Happiness])
   
s_happiness = State(Happiness, 'happiness status')
s_sunny = State(day, 'sunny day')
s_raise = State(day, 'raise day')
model = BayesianNetwork('happiness status')
model.add_states(s_happiness, s_sunny, s_raise)
model.add_transition(s_sunny, s_happiness)
model.add_transition(s_raise, s_happiness)
model.bake() # finalize the topology of the model

#print(model)
#print()

print ('The number of nodes:', model.node_count())
print ('The number of edges:', model.edge_count())

# predict_proba(Given factors)
print()

# part a - P(o2|g, ~o1)
print (model.predict_proba({'rainy':'happiness'})[1].parameters)
print()

# part b - P(g|o1, o2)
print (model.predict_proba({'happiness':'raise','sunny':'raise'})[0].parameters)
print()

# part c
print (model.predict_proba({'raise':'happiness'})[0].parameters)
print()

# part d
print (model.predict_proba({'hapiness':'raise','raise':'raise'})[0].parameters)
print()