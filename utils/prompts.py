FEW_SHOT_PROMPT = """###INSTRUCTIONS###

You must follow the rules before answering:
- A question and its answer options will be provided.
- There can be multiple correct options.
- The correct answer is always given.

###EXAMPLES###

Question: 'Which of the Mass Effect games is playable on the Wii U?'
Options:
0. {"answer": "Mass Effect: Andromeda", "WikiDataID": "Q20113552"}
1. {"answer": "Mass Effect: Andromeda", "WikiDataID": "Q96392210"}
2. {"answer": "Mass Effect Andromeda: Annihilation", "WikiDataID": "Q95727871"}
3. {"answer": "Mass Effect Andromeda: Annihilation", "WikiDataID": "Q97319062"}
4. {"answer": "Mass Effect 3: From Ashes", "WikiDataID": "Q51929525"}
5. {"answer": "Mass Effect 3: Citadel", "WikiDataID": "Q51929681"}
6. {"answer": "Mass Effect 3", "WikiDataID": "Q753511"}
7. {"answer": "Mass Effect 2: Overlord", "WikiDataID": "Q16267640"}
8. {"answer": "Mass Effect 2: Lair of the Shadow Broker", "WikiDataID": "Q6223902"}
9. {"answer": "Mass Effect 2", "WikiDataID": "Q725057"}
Answer: 6

Question: 'Who has assisted more times, David Beckham or Ronaldinho?'
Options:
0. {"answer": "trial of Lionel and Jorge Messi", "WikiDataID": "Q41496125"}
1. {"answer": "Ronaldo", "WikiDataID": "Q529207"}
2. {"answer": "Ronaldo", "WikiDataID": "Q19819804"}
3. {"answer": "Ronaldinho", "WikiDataID": "Q106547625"}
4. {"answer": "Ronaldinho", "WikiDataID": "Q39444"}
5. {"answer": "Pepe Garza", "WikiDataID": "Q37846050"}
6. {"answer": "Lionel Messi", "WikiDataID": "Q615"}
7. {"answer": "David Beckham Academy", "WikiDataID": "Q5231237"}
8. {"answer": "David Beckham", "WikiDataID": "Q10520"}
9. {"answer": "Cristiano Ronaldo", "WikiDataID": "Q5835713"}
10. {"answer": "Cristiano Ronaldo", "WikiDataID": "Q11571"}
Answer: 8

###Answering Rules###

1. For each correct option, write only its digit.
2. If you do not know the answer, propose the most likely ones.

###QUESTION###

"""