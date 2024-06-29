# This file takes in a dataset of "math questions" (i.e. load_dataset('gsm8k', 'main')['train']) and creates programs out of them you can call infinitely later. (note I already did this for GSM8k, so you don't have to.. check the repo)
# Be careful, as it will cost $$ to ping OAI.

import json
import openai
import time
import pprint
import re
import numpy as np
from datasets import load_dataset
from tqdm import tqdm



def text_generator_positive(program_text,n_examples=5):
    examples = []
    # validate the program generated
    try:
        exec(program_text, globals())
    except Exception as e:
        output = f"Error executing the generated program: {e.stderr}"
        print(output)
        return None
    
    for i in range(n_examples):
        try:
            example = generate_math_problem() # the default name of the program in the program_text string
            examples.append(example)
        except Exception as e:
            output = f"Error executing the generated program: {e.stderr}"
            print(output)
        
    return examples

def create_program_batch_file(train_dataset,batch_file_path = "./batchinput.jsonl")
    # Take in train_dataset=[{'question':<>, 'answer':<>}, ... , ] i.e. dataset = load_dataset('gsm8k', 'main')['train']
    # Output a batch_file to upload to openai
    with open(batch_file_path, "w") as f:
        for i,y in enumerate(train_dataset):
            prompt = {
                "custom_id": f"request_{i+1}",  # Add custom_id to uniquely identify each request
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4",
                    "messages": [
                                    {
                                        "role": "user",
                                        "content": '''You are an expert python programmer, and you are assigned this ticket. You must take this "example math problem", 
                                                and make a python program that will randomly generate the numbers in the math problem but keep the same formatting as the original example math problem, 
                                                but still gives the right answers in the answer section. Your function should be called generate_math_problem() and it should return a dict = \{"question": question, "answer": answer\}
                                                Your generated math problem should follow the EXACT format and conventions of the "example math problem" provided below. Not any other text.
                                                Your program should not start with \'\'\'python or anything of the sort. 
                                                Follow all conventions of the example math problem. 
                                                YOU MUST WRAP YOUR OUTPUT CODE IN YOUR RESPONSE IN <code> </code> BLOCKS LIKE SO: <code>def f(): return 7</code> and we will extract all plain-text in between the code blocks and compile it directly.
                                                DO NOT USE ANY "LINE CONTINUATION CHARACTERS" IN YOUR PROGRAM NO MATTER HOW LONG THE LINE IS IN YOUR PROGRAM! 

                                                example math problem = ''' + str(y),
                                    }
                                ]
                            }
            }
            f.write(json.dumps(prompt) + "\n")


# Example how to use this:
if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "<INSERT YOUR KEYS>"
    batch_file_path = "./batchinput.jsonl" # temp file to upload to oai
    output_file = './output_for_programs.txt' #where oai passed..
    bad_program_output_file = './bad_output_for_programs.txt' #where oai failed..


    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


    # Initialize the OpenAI client
    client = openai.OpenAI()

    # Create the batch input file 
    create_program_batch_file(train_dataset,batch_file_path = batch_file_path )

    # Upload the batch file to chatgpt
    batch_input_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    # Submit the batch job... note this cost me about $200 on OAI! Be careful. Could be much cheaper if you use their API in a smarter way but I am lazy.
    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )

    batch_job_id = batch_job.id
    print(f"Batch job submitted. ID: {batch_job_id}")


    # Function to check the status of the batch job
    def check_batch_status(batch_id):
        batch_status = client.batches.retrieve(batch_id)
        return batch_status

    # Periodically check the status of the batch job
    while True:
        status = check_batch_status(batch_job_id)
        print(f"Batch job status: {status.status}")
        if status.status == "completed":
            break
        elif status.status in ["failed", "expired"]:
            print(f"Batch job failed or expired: {status}")
            break
        time.sleep(10)  # Wait for a minute before checking again

    # Once completed, download it, and open it up..
    if status.status == "completed":
        # Retrieve the output file ID from the completed batch job
        output_file_id = status.output_file_id

        # Get the content of the output file
        output_file_content = client.files.content(output_file_id)

        # Read and print the output content
        output_content = output_file_content.read().decode("utf-8")
        print("Output file content:")
    #     print(output_content)
    else:
        print("Batch job did not complete successfully.")



    # Define the start and end strings to parse out of this.. ALSO DUMB BUT I AM LAZY! :) 
    string_start = r'{"role": "assistant", "content": '
    string_end = r'},\s*"\s*logprobs": null, "finish_reason": "stop"}],'

    # Use a regular expression to find all instances of content between start and end
    pattern = re.compile(f'{string_start}(.*?){string_end}', re.DOTALL)
    matches = pattern.findall(output_content)

    # Clean up the extracted content
    list_of_programs = [match.replace('\\"','"').replace("\\n","\n").replace('\\\n', ' ').replace("\\", '').strip('"') for match in matches]

    # Go through each one and make sure OAI did what you wanted.. about 10% of the time it doesn't do it.. you could sample a few times but again, lazy... 
    list_of_programs_validated = []
    list_of_programs_not_validated = []
    failed_counter=0
    for program in list_of_programs:
        pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
        matches = pattern.findall(program)
        if len(matches)!=1:
            print(f"matches not equal to 1, gpt not following code block pattern correctly: {matches}")
            failed_counter+=1
            list_of_programs_not_validated.append(program)
            continue
            
        program_text = matches[0].strip()
        
        try:
            exec(program_text)
            list_of_programs_validated.append(program_text)
        except Exception as e:
            print("failed: "+str(failed_counter)+" "+ program_text)
            failed_counter+=1
            list_of_programs_not_validated.append(program)
        
    list_of_programs_not_validated_still_not_validated = []
    for j in list_of_programs_not_validated:
        
        pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
        matches = pattern.findall(j)
        if len(matches)==1:
            program_text = matches[0]

            try:
                exec(program_text)
                list_of_programs_validated.append(program_text)
                continue
            except Exception as e:
                print("failed: "+str(failed_counter)+" "+ program_text)
                failed_counter+=1

        list_of_programs_not_validated_still_not_validated.append(j)

    print("list_of_programs_not_validated_still_not_validated" + str(len(list_of_programs_not_validated_still_not_validated)) )
    print("list_of_programs_not_validated" + str(len(list_of_programs_not_validated))  )
    print("list_of_programs_validated" + str(len(list_of_programs_validated)) )


    # now lets create a file of all programs, that we will create once, and then save off locally.
    for program in tqdm(list_of_programs_validated):
        with open(output_file, "a") as f:
            for k in program.split("\n"):
                f.write(k + '\n')
            f.write('-----------<this will be used to split on later>----------------\n')

    for program in tqdm(list_of_programs_not_validated_still_not_validated):
        with open(bad_program_output_file, "a") as f:
            for k in program.split("\n"):
                f.write(k + '\n')
            f.write('-----------<this will be used to split on later>----------------\n')
    

    # now you can read it in during training like this:

    # Define the delimiter
    delimiter = "-----------<this will be used to split on later>----------------"

    # Initialize an empty list to store the programs
    list_of_programs = []

    # Read the file
    with open(output_file, 'r') as file:
        # Read the entire content of the file
        content = file.read()
        
        # Split the content based on the delimiter
        list_of_programs = content.split(delimiter)

    # Strip any leading or trailing whitespace from each program
    list_of_programs = [program.strip() for program in list_of_programs]
    print('# programs: '+str(len(list_of_programs)))
    
    # Now you can call all of the programs like so:
    for program_text in list_of_programs:
        print(text_generator_positive(program_text,n_examples=5))