# sentence_augs

Inspired by the fantastic library imaug (https://imgaug.readthedocs.io/en/latest/) there should be a way to "explode" a train set of sentences (input-output pairs) to make millions more augmented by a functions that will likely happen at test time to improve generalization. 

--------------------------------------------------------------------------------
Grammar/Logic Lossless augs:
--------------------------------------------------------------------------------

- swap_with_synonym (find all verbs / adjectives, and loop over and look up synonymns and replace.. i.e. Suzy is a "mean" person. She "yelled at" Greg yesterday. -> Suzy is a[ill-mannered,impolite,discourteous,impertinent,insolent,impudent,cheeky,audacious,presumptuous,uncivil,unpleasant,disagreeable,nasty,harsh] person. She [berated, castigated, chewed out, dressed down, punished, reprimanded, ripped into, scolded, told off] Greg yesterday.
)
- swap_pronouns (find all pronouns, and loop over and find all and replace.. i.e. Suzy likes to bike. Suzy likes to bike with Greg sometimes, Greg does not like biking with Suzy. -> Dan likes to bike. Dan likes to bike with Greg sometimes, Greg does not like biking with Dan.)
- turn_into_program_and_sample_randomly: (take GSM8k for example, take each "math problem" turn it into a program and then make infinite examples of it so your model won't overfit to the text and will abstract further to learn to solve the actual problem...) i.e.

-- {'question': 'Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $617 and spent $44 on a button-up shirt, $39 on suit pants, $45 on a suit coat, $56 on socks, and $47 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $39 left from her budget. How much did Alexis pay for the shoes?', 

-- 'answer': 'Let S be the amount Alexis paid for the shoes.\nShe spent S + 44 + 39 + 45 + 56 + 47 = S + <<+44+39+45+56+47=231>>231.\nShe used all but $39 of her budget, so S + 231 = 617 - 39 = 578. Thus, Alexis paid S = 578 - 231 = $<<578-231=347>>347 for the shoes.\n#### 347'} 


-- we want to turn that into a program like so:

def generate_math_problem():
    budget = random.randint(100, 1000)
    shirt_cost = random.randint(10, budget//6)
    pants_cost = random.randint(10, budget//6)
    coat_cost = random.randint(10, budget//6)
    socks_cost = random.randint(1, budget//10)
    belt_cost = random.randint(5, budget//8)
    left_over_money = random.randint(1, budget//10)
    shoes_cost = budget - (shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost + left_over_money)
    sum_of_costs_without_shoes = shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost
    total_spent_without_left_over = budget - left_over_money

    question = f'Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of ${budget} and spent ${shirt_cost} on a button-up shirt, ${pants_cost} on suit pants, ${coat_cost} on a suit coat, ${socks_cost} on socks, and ${belt_cost} on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has ${left_over_money} left from her budget. How much did Alexis pay for the shoes?'

    answer = f'Let S be the amount Alexis paid for the shoes.\nShe spent S + {shirt_cost} + {pants_cost} + {coat_cost} + {socks_cost} + {belt_cost} = S + <<+{shirt_cost}+{pants_cost}+{coat_cost}+{socks_cost}+{belt_cost}={sum_of_costs_without_shoes}>>{sum_of_costs_without_shoes}.\nShe used all but ${left_over_money} of her budget, so S + {sum_of_costs_without_shoes} = {budget} - {left_over_money} = {total_spent_without_left_over}.\nThus, Alexis paid S = {total_spent_without_left_over} - {sum_of_costs_without_shoes} = $<<{total_spent_without_left_over}-{sum_of_costs_without_shoes}={shoes_cost}>>{shoes_cost} for the shoes.\n#### {shoes_cost}'

    return {"question": question, "answer": answer}


-- Then we can call it infinitely.

--------------------------------------------------------------------------------
Grammar Lossy augs (useful for reward modeling):
--------------------------------------------------------------------------------

- addword (add a random word into a seq..)
- subtractword  (sub a random word from a seq..)
- swapwords  (swap random words in a seq..)
- addline  (add random line into a seq..)
- subtractline  (sub a random line in a seq..)
- addchar  (add a random char into a seq..)
- subchar  (sub a random char in a seq..)
- swapnumbers  (take a random number in the seq, and then make it a new number in a seq..)
- addspaces   (add random spaces to the seq)


--------------------------------------------------------------------------------
# Example how to use this:
--------------------------------------------------------------------------------


```

model_id = "EleutherAI/pythia-70m-v0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id)

user_prompt = "User: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

target_response = "Response: Let's break this down step by step.\n\n**Step 1: Understand the problem**\nNatalia sold clips to 48 friends in April. Then, she sold half as many clips in May. We need to find the total number of clips she sold in April and May.\n\n**Step 2: Calculate the number of clips sold in May**\nIf Natalia sold half as many clips in May as she did in April, that means she sold:\n\n48 (clips sold in April) / 2 = 24 clips in May\n\n**Step 3: Add the number of clips sold in April and May**\nTo find the total number of clips sold, we add the number of clips sold in April and May:\n\n48 (clips sold in April) + 24 (clips sold in May) = 72\n\n**Conclusion**\nNatalia sold a total of 72 clips in April and May."

string_true = user_prompt + '\n<sep>\n' + target_response

## negative sampling (for reward modeling)
for i in range(20):
    print("="*50)
    print("="*50)
    string_corrupted = text_corrupter_negative(target_response)
    r_mask_truth = generate_match_mask(tokenizer, target_response, string_corrupted)
    print(string_true)
    print("="*50)
    print(string_corrupted)
    print(r_mask_truth)

```