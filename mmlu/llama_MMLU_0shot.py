import json
import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset
from langchain_ollama.llms import OllamaLLM
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Initialize the model
model = OllamaLLM(
        model="llama3:8b",
        num_gpu=1,
        temprature=0,
        verbose=1
)



response_schemas = [
	ResponseSchema(name="answer", description="answer to the Question."),
	ResponseSchema(name="Option", description="The correct option. The output of this field should only be in A, B, C or D Nothing else. Remember that!"),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)


# Choices
MMLU_CHOICES = ['A', 'B', 'C', 'D']

# Prompt
Template = """
Answer the following multiple choice question enclosed in the triple backticks about the {formatted_subject}.
\n
```
{question}
\n
{choices}
```

\n
\n
{format_instructions}

"""


prompt = PromptTemplate(
        template = Template,
        input_variables = ["question", "formatted_subject", "examples", "choices"],
        partial_variables = {"format_instructions" : parser.get_format_instructions()}
)


mmlu_dataset = load_dataset(
        path = 'cais/mmlu', 
        name = 'all', 
        trust_remote_code=True, 
        split = 'test'
)


df_mmlu = mmlu_dataset.to_pandas()

subjects = sorted(df_mmlu['subject'].value_counts().keys())

def formatChoices(all_choices):
        choices = ''
        for index, choice in enumerate(all_choices):
                choices += f'{MMLU_CHOICES[index]}- {choice}.\n'
        return choices

# Langchian Chain
chain_parse = prompt | model | parser

Overall_score = []
Task = []
Score = []
all_is_corrects = []

# iterating over subjects
for i, subject in enumerate(subjects):


	df_subject = df_mmlu[df_mmlu['subject'] == subject]
	formatted_subject = subject.replace('_', ' ')

        # instantiating containers to old all boolean "is correct" values and probabilities
	subject_is_correct =  []

	print(f"\n\n\n\t\t{formatted_subject}\n---------------------------------------------------")


        # iterating over all the rows of the DataFrame
	for index, row in df_subject.iterrows():

                # Extracting the question from row
		question = row['question']

                # Extracting the choices from the row
		choices = formatChoices(row['choices'])

                # Extracting the answrs from the row
		number_answer = row['answer']
		ground_truth = MMLU_CHOICES[number_answer]


		try:
	                # Prompting langchain chain
			response = chain.invoke({
				formatted_subject : formatted_subject,
				question : question,
				choices : choices
			})
			print("\n", end=" ")
			# Checking prediction
			is_true = response['Option'] == ground_truth
			print(response['Option'], end=" ")

		except:
			temp = prompt.format(
				formatted_subject = formatted_subject,
                	        question = question,
                	        choices = choices
			)

			resp = model.invoke(temp)

			tmp_prompt = "What is the correct option chosen in the llm response enclosed in the triple backticks.\n\njust return the correct option chosen(A, B, C or D). Give me a single alphabhet output(A, B, C, D), print nothing else.\n\nresponse:```{re}\nchoices:{choices}```"
			prediction = model.invoke(tmp_prompt.format(re = resp, choices=choices))
			is_true = prediction == ground_truth
			print(prediction, end=". ")


                # Append The final values
		subject_is_correct.append(is_true)
		Overall_score.append(is_true)

        # Appending the mean of all the correct subject answer to the full list of correct answer
	all_is_corrects.append(np.array(np.mean(subject_is_correct)))
	print("Evaluated", subject, f"Score: {all_is_corrects[-1]}")
	Score.append(all_is_corrects[-1])
	Task.append(subject)



weighted_acc = np.mean(all_is_corrects)
print("OverAll subject score", weighted_acc)

Accuracy = np.mean(Overall_score)

Score.append(weighted_acc)
Task.append("Overall Subject Score")

Score.append(Accuracy)
Task.append("Accuracy")

print(f"Accuracy: {Accuracy}")
data = pd.DataFrame({
	"Task" : Task,
	"Score" : Score
})

data.to_csv("MMLU_llama3_8b_0shot.csv", index=False)


