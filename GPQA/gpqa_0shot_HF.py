# Use a pipeline as a high-level helper
import transformers
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
	"text-generation",
	model=model_id, 
	model_kwargs={"torch_dtype": torch.bfloat16}, 
	device="cuda:1"
)

#print(pipeline("What is Hugging Face?"))

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_main.csv")


tags = ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']
CHOICES = ['A', 'B', 'C', 'D']

response_schemas = [
        ResponseSchema(
	name="answer",
	description="Correct Answer to the Question."
	),
	ResponseSchema(
	name="Explanation",
	description="Justify the Answer chosen in a descriptive manner. Make it as short as possible."
	),
        ResponseSchema(
	name="Option",
	description = "Index of the correct option. This field should only contain A, B, C or D, nothing else."
	)
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)


prompt_template = """
Answer the following multiple choice question enclosed in the triple backticks about {subject}.
\n
```
{question}
\n
{choices}
```

\n
\n

{format_instructions}

Answer:
"""

prompt = PromptTemplate(
        template = prompt_template,
        input_variables = ["question", "subject", "choices"],
        partial_variables = {"format_instructions" : parser.get_format_instructions()}
)


subjects = sorted(df['Subdomain'].value_counts().keys())
Task = []
subject_score = []
all_is_correct = []
Overall_score = []

# iterate over each subject
for id, subject in enumerate(subjects):

	df_subject = df[df["Subdomain"] == subject]
	subject_is_correct = []

	print(f"\n\n\n\t\t{subject}\n---------------------------------------------------")

	for index, row in df_subject.iterrows():

		question = row["Question"]

		choices = ''
		for i, tag in enumerate(tags):

			choices += f'{CHOICES[i]}-    {row[tag]}\n'

		ground_truth = CHOICES[0]


		message = prompt.format(
		question = question,
		choices = choices,
		subject = subject
		)

		response = pipeline(message)
		split_response = response[0]['generated_text'].split("Answer:")[-1]

		try:

			prediction = parser.parse(split_response)

			predicted_option = prediction["Option"]
			is_true = predicted_option[0] == ground_truth
			print(predicted_option[0], end=" ")

		except:

			predicted_option = split_response.split("\"Option\": \"")[-1][0]
			is_true = predicted_option == ground_truth
			print(predicted_option, end=" ")

		subject_is_correct.append(is_true)
		Overall_score.append(is_true)

	all_is_correct.append(np.array(np.mean(subject_is_correct)))
	print(f"Evaluated {subject} Score: {all_is_correct[-1]}")
	Task.append(subject)
	subject_score.append(all_is_correct[-1])



accuracy = np.mean(all_is_correct)
print(f"Overall Score: {accuracy}")

print(f"Accuracy(indivisual mean): {np.mean(Overall_score)}")

subject_score.append(accuracy)
Task.append("Subject mean Score")

subject_score.append(np.mean(Overall_score))
Task.append("Overall Accuracy")


data = pd.DataFrame({
	"Subdomain" : Task,
	"Score:" : subject_score
})

data.to_csv("Data/GPQA_llama3_8b_0shot_HF.csv", index=False)








