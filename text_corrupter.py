import random
import torch
from transformers import GPTNeoXModel, GPTNeoXForCausalLM, GPTNeoXConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from copy import deepcopy
from typing import Optional, Tuple, Union
from torch import nn, optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
from typing import Optional, Tuple, Union
import random
import numpy as np

def text_corrupter_negative(y):
    # main function for "grammar lossy" sampling
    
    addword=True
    subtractword=True 
    swapwords=True 
    addline=True
    subtractline=True 
    addchar=True
    subtractchar=True
    swapnumbers=True
    addspaces=True 
    
    y_lines = [i+'\n' for i in y.split('\n')]
    y_words_with_lines = []
    
    for line in y_lines:
        y_words_with_lines.extend(line.split() + ['\n'])

    random_words = y_words_with_lines + ["example", "corrupt", "random", "text", "sample", "hello"]
    
    random_lines = y_lines + ["This is a random line.", "Another example of a random line.", "Yet another.", "blah", ""]
    
    random_chars = [chr(i) for i in range(128)]
    
    def add_random_char(words):
        if not words:
            return words, None
        position = random.randint(0, len(words) - 1)
        char_position = random.randint(0, len(words[position]))
        words[position] = words[position][:char_position] + random.choice(random_chars) + words[position][char_position:]
        return words, (position, char_position)
    
    def subtract_random_char(words):
        if not words:
            return words, None
        position = random.randint(0, len(words) - 1)
        if not words[position]:
            return words, None
        char_position = random.randint(0, len(words[position]) - 1)
        words[position] = words[position][:char_position] + words[position][char_position + 1:]
        return words, (position, char_position)
    
    def add_random_word(words):
        random_word = random.choice(random_words)
        position = random.randint(0, len(words))
        words.insert(position, random_word)
        return words, position

    def subtract_random_word(words):
        if not words:
            return words, None
        position = random.randint(0, len(words) - 1)
        words.pop(position)
        return words, position

    def add_random_line(lines):
        random_line = random.choice(random_lines)
        position = random.randint(0, len(lines))
        lines.insert(position, random_line)
        return lines, position

    def subtract_random_line(lines):
        if not lines:
            return lines, None
        position = random.randint(0, len(lines) - 1)
        lines.pop(position)
        return lines, position

    def swap_random_words(words):
        if len(words) < 2:
            return words, None
        pos1, pos2 = random.sample(range(len(words)), 2)
        words[pos1], words[pos2] = words[pos2], words[pos1]
        return words, (pos1, pos2)

    def swap_numbers(words):
        positions = [i for i, word in enumerate(words) if any(char.isdigit() for char in word)]
        if len(positions) < 2:
            return words, None
        pos1, pos2 = random.sample(positions, 2)
        words[pos1], words[pos2] = words[pos2], words[pos1]
        return words, (pos1, pos2)

    def add_random_spaces(words):
        position = random.randint(0, len(words) - 1)
        spaces = ' ' * random.randint(1, 5)
        words[position] = words[position] + spaces
        return words, position

    operations = []
    if addword:
        operations.append("addword")
    if subtractword:
        operations.append("subtractword")
    if addline and len(y_lines) > 1:
        operations.append("addline")
    if subtractline and len(y_lines) > 1:
        operations.append("subtractline")
    if swapwords:
        operations.append("swapwords")
    if addchar:
        operations.append("addchar")
    if subtractchar:
        operations.append("subtractchar")
    if swapnumbers:
        operations.append("swapnumbers")
    if addspaces:
        operations.append("addspaces")
        
    if not operations:
        return y

    operation = random.choice(operations)
    print("chosen operation is: " + operation)

    if operation == "addword":
        y_words_with_newlines, pos = add_random_word(y_words_with_lines)
        
    elif operation == "subtractword":
        y_words_with_newlines, pos = subtract_random_word(y_words_with_lines)
        
    elif operation == "addline":
        y_lines, pos = add_random_line(y_lines)
        
    elif operation == "subtractline":
        y_lines, pos = subtract_random_line(y_lines)
        
    elif operation == "swapwords":
        y_words_with_newlines, (pos1, pos2) = swap_random_words(y_words_with_lines)
        
    elif operation == "addchar":
        y_words_with_newlines, pos = add_random_char(y_words_with_lines)
        
    elif operation == "subtractchar":
        y_words_with_newlines, pos = subtract_random_char(y_words_with_lines)
        
    elif operation == "swapnumbers":
        y_words_with_newlines, (pos1, pos2) = swap_numbers(y_words_with_lines)
        
    elif operation == "addspaces":
        y_words_with_newlines, pos = add_random_spaces(y_words_with_lines)

    if operation == "addline" or operation == "subtractline":
        y_corrupted = ''.join(y_lines)
    else:
        y_corrupted = ' '.join(y_words_with_newlines)
        y_corrupted = y_corrupted.replace(' \n', '\n').replace('\n ', '\n') 
        

    return y_corrupted

def generate_match_mask(tokenizer, string_true, string_corrupted):
    # Generate a mask for reward sampling that will be 0 where the aug occurred, and 1 where it matches.

    # Tokenize the input strings
    tokens_true = tokenizer.tokenize(string_true)
    tokens_corrupted = tokenizer.tokenize(string_corrupted)
    
    # Initialize the match mask with zeros
    match_mask = [0] * len(tokens_true)
    
    # Calculate the minimum length to avoid index out of range
    min_length = min(len(tokens_true), len(tokens_corrupted))
    
    # Compare tokens and generate the match mask
    for i in range(min_length):
        if tokens_true[i] == tokens_corrupted[i]:
            match_mask[i] = 1
    
    return match_mask

    