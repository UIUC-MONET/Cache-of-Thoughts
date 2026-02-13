import asyncio
import os
from openai import AsyncOpenAI

class ProgressLog:
    def __init__(self, total, task_name=""):
        self.total = total
        self.done = 0
        self.task_name = task_name
        

    def increment(self):
        self.done = self.done + 1

    def __repr__(self):
        return f"<openai> {self.task_name}: done runs {self.done}/{self.total}."


async def get_completion(content, semaphore, progress_log):
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    async with semaphore:
        await asyncio.sleep(1)
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model="gpt-4o",
        )
        progress_log.increment()
        print(progress_log)
        return chat_completion.choices[0].message.content


async def get_completion_list(content_list, max_parallel_calls):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)
    progress_log = ProgressLog(len(content_list))

    return await asyncio.gather(*[get_completion(content, semaphore, progress_log) for content in content_list])
    
    
async def get_vision_completion(img_base64: str, prompt: str, semaphore, progress_log, model: str="gpt-4o"):
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    async with semaphore:
        await asyncio.sleep(1)
        try:
            chat_completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                model=model,
            )
            progress_log.increment()
            print(progress_log)
            return chat_completion
        except Exception as e:
            print(e)
            return None

async def get_vision_completion_list(img_base64_list, prompt_list, max_parallel_calls, model="gpt-4o", task_name=""):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)
    progress_log = ProgressLog(len(prompt_list), task_name=task_name)

    return await asyncio.gather(*[get_vision_completion(img_base64, prompt, semaphore, progress_log, model=model) 
                                  for img_base64, prompt in zip(img_base64_list, prompt_list)])

async def get_vision_completion_multi_image_list(img_base64_list_of_list, prompt_list, max_parallel_calls, model="gpt-4o", task_name=""):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)
    progress_log = ProgressLog(len(prompt_list), task_name=task_name)

    return await asyncio.gather(*[get_vision_completion(img_base64_list, prompt, semaphore, progress_log, model=model) 
                                  for img_base64_list, prompt in zip(img_base64_list_of_list, prompt_list)])