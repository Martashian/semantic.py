# Create a file called semantic.py and run all the code extracts from the lesson
import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 =nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))


tokens = nlp("cat apple monkey banana")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go", "Hello, there is my car", "I\'ve lost my car in my car", "I\'d like my boat back", "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# Write a note about what you found interesting about the similarities between cat, monkey and banana and think of an example of your own
'''I found interesting that the similarities between cat and monkey were the highest ones due to the fact that both are animals, 
and not the similarities banana - monkey considering the popular knowledge that monkeys like to eat bananas.'''

'''An example of my own could be adding elephant and peanut to the comparision. Monkey and elephant have the highest similarity, even more
than cat and monkey, I assume it is because both monkeys and elephants are wild animals, unlike cats. The second highest similarity
is between banana and peanut, both food.'''
word4 = nlp("elephant")
word5 = nlp("peanut")

print(word3.similarity(word4)) # banana elephant
print(word3.similarity(word5)) # banana peanut
print(word2.similarity(word5)) # monkey peanut
print(word2.similarity(word4)) # monkey elephant
print(word4.similarity(word5)) # elephant peanut

# Run the example file with the simpler language model "en_core_web_sm" and write a note on what you notice is different
# from the model "en_core_web_md"
'''When I run the example file with en_core_web_sm I notice that a warning message appears letting me know
that this model does not have word vectors loaded and that the result of the Doc.similarity method may
not give useful similarity judgements. "en_core_web_md" does not encounter any issues and accurately prints
similarities.'''
