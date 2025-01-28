flamingo:

mmmu first prompt:
<image><question> The options are the following:<option> There is only one option possible.The answer is

mmmu n-shot cache sample:
Human: <question> <option>. Please include your reasoning steps, then answer your choice in this format: ANSWER: LETTER CHOICE. The letter choice is strictly in the alphabetical order, and there is only one option possible. Assistant: <gpt4o-response>

mmmu second prompt:
<image><shot-1><|endofchunk|>...<image><shot-n><|endofchunk|><image><question> The options are the following:<option> There is only one option possible.The answer is

clevr first prompt:
<image>How many <color|material|shape|size> objects?The answer is

clevr n-shot cache sample:
Human: How many objects in the image have the <adj><color|material|shape|size> Please include your reasoning steps, then answer your choice in this format: ANSWER: NUMBER. Assistant: <gpt4o-response>

clevr second prompt:
<image><shot-1><|endofchunk|>...<image><shot-n><|endofchunk|><image>How many <color|material|shape> objects?The answer is

textocr first prompt:
<image>What text is shown in the red box?Only answer with the largest text.The answer is

textocr n-shot cache sample:
Human: What text is shown in the red box? Only answer with the largest text. Please include your reasoning steps, then answer your choice in this format: ANSWER: TEXT. Assistant: <gpt4o-response>

textocr second prompt:
<image><shot-1><|endofchunk|>...<image><shot-n><|endofchunk|><image>What text is shown in the red box?Only answer with the largest text.The answer is

qwen:

textocr first prompt:
Now you should answer the following question:
Human:
An image will be provided where a red box is drawn around the text of interest. Answer with the largest text inside the red box. Ensure that the transcription is precise, reflecting the exact characters, including letters, numbers, symbols.
Assistant(you):

textocr n-shot example:
Example n
Human: What text is shown in the red box? Only answer with the largest text. Please include your reasoning steps, then answer your choice in this format: ANSWER: <TEXT>.
Assistant: <gpt4o-response>

text second prompt:
This is a chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. The assistant will have one similar in-context example provided by another powerful assistant:
<shot-1>...<shot-n>
Now you should answer the following question given the image below and you can use GPT4 Assistant's case for reference:
Human:
An image will be provided where a red box is drawn around the text of interest. Answer with the largest text inside the red box. Ensure that the transcription is precise, reflecting the exact characters, including letters, numbers, symbols.
Assistant(you):



<first task description> Now you should answer the following question:
<textocr question> An image will be provided where a red box is drawn around the text of interest. Answer with the largest text inside the red box. Ensure that the transcription is precise, reflecting the exact characters, including letters, numbers, symbols.
<n-shot example> Example n
Human: What text is shown in the red box? Only answer with the largest text. Please include your reasoning steps, then answer your choice in this format: ANSWER: <TEXT>.
Assistant: <gpt4o-response>
<n-shot description> This is a chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. The assistant will have one similar in-context example provided by another powerful assistant:
<second task description> Now you should answer the following question given the image below and you can use GPT4 Assistant's case for reference:

textocr first prompt:
<first task description>
Human:
<textocr question>
Assistant(you):

textocr second prompt:
<n-shot description>
<n-shot example>
:
<n-shot example>
<second task description>
Human:
<textocr question>
Assistant(you):

<clevr question>
How many objects in the image have the rubber material. Please answer in the following format: ANSWER: <NUMBER>
<n-shot example> Example n
Human: How many objects in the image have the <adj> <color|material|shape|size> Please include your reasoning steps, then answer your choice in this format: ANSWER: <NUMBER>.
Assistant: <gpt4o-response>

<mmmu question>
Human: 
<question> <option>. Please include your reasoning steps, then answer your choice in this format: ANSWER: <LETTER CHOICE>. The letter choice is strictly in the alphabetical order, and there is only one option possible.
<n-shot example>
Example n
Human: <question> <option> Please include your reasoning steps, then answer your choice in this format: ANSWER: <LETTER CHOICE>. The letter choice is strictly in the alphabetical order, and there is only one option possible.
Assistant: <gpt4o-response>
