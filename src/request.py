import os
import json
import time
import openai
import logging
import aiohttp
import asyncio
from dataclasses import dataclass, field
import shutil
from tqdm import tqdm
from src.tokens import num_tokens_from_messages

logger = logging.getLogger(__name__)

async def async_api_requests(
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    request_url: str,
    api_key: str, 
    root_path:str,
    result_file_path: str, 
    result_file_name: str, 
    task: str,
    dataset: str, 
    model: str ='gpt-4o-mini',
    dataNum: int =0, 
    testNum: int =1, 
    method: str ='base', 
    max_token: int =8000, 
    max_attempts: int =10,
    temperature: float = 0,
    choices: int = 1,
    data = None, 
    ):
    
    # Constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.01
    checkpoint_interval = 1000  # <--- SAVE EVERY 1000 ITEMS
    last_saved_count = 0

    request_header = {"Authorization": f"Bearer {api_key}"}
    queue_of_requests_to_retry = asyncio.Queue()
    status_tracker = StatusTracker()
    next_request = None

    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    not_finished = True
    
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)
    results_json_file = os.path.join(result_file_path, result_file_name + ".json")

    results_list = []

    logging.basicConfig(format='%(asctime)s %(message)s', filename=os.path.join(result_file_path, result_file_name+".log"), encoding='utf-8', level='WARNING')

    openai.api_key = api_key
    testNum = min(testNum, len(data))
    global pbar
    pbar = tqdm(total = testNum-dataNum) 

    while(True):
        if next_request is None:
            if not queue_of_requests_to_retry.empty(): 
                next_request = queue_of_requests_to_retry.get_nowait()
            elif (not_finished):
                if dataNum < testNum:                    
                    request_id = data[dataNum]['id']
                    messages = data[dataNum]['prompt']
                    request_truth = data[dataNum]['ground_truth']
                    
                    request_json = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": 1,
                        "n": choices,
                        "stream": False,
                    }
                    next_request = APIRequest(
                        request_id=request_id,
                        request_json=request_json,
                        request_truth=request_truth,
                        token_consumption=num_tokens_from_messages(messages, model),
                        attempts_left=max_attempts,
                        metadata=request_json.pop("metadata", None),
                        results_list=results_list,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    dataNum += 1
                else:
                    not_finished = False

        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0, max_requests_per_minute)
        available_token_capacity = min(available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0, max_tokens_per_minute)
        last_update_time = current_time

        if next_request:
            next_request_tokens = next_request.token_consumption
            if available_request_capacity >= 1 and available_token_capacity >= next_request_tokens:
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                asyncio.create_task(
                    next_request.call_api(
                        request_url=request_url,
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        save_filepath=results_json_file,
                        status_tracker=status_tracker,
                    )
                )
                next_request = None

        # --- CHECKPOINT SAVING LOGIC ---
        current_count = len(results_list)
        if current_count - last_saved_count >= checkpoint_interval:
            write_file(results_list, results_json_file)
            last_saved_count = current_count
            # Log checkpoint to console without breaking TQDM flow
            tqdm.write(f"Checkpoint saved: {current_count} items currently processed.")

        if status_tracker.num_tasks_in_progress == 0 and not not_finished:
            break

        await asyncio.sleep(seconds_to_sleep_each_loop)

        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            await asyncio.sleep(seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)

    # Final Save to ensure the last batch (the remainder of 1000) is written
    write_file(results_list, results_json_file)
    pbar.close()

@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0

@dataclass
class APIRequest:
    request_id: int
    request_json: dict
    request_truth: str
    token_consumption: int
    attempts_left: int
    metadata: dict
    results_list: list
    result: list = field(default_factory=list)

    async def call_api(self, request_url, request_header, retry_queue, save_filepath, status_tracker):
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url=request_url, headers=request_header, json=self.request_json) as response_raw:
                    response = await response_raw.json()
            if "error" in response:
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1
        except Exception as e:
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left > 0:
                retry_queue.put_nowait(self)
            else:
                result = {'id': self.request_id, 'ground_truth': self.request_truth, 'prompt': self.request_json, 'response': str(error)}
                self.results_list.append(result)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
                pbar.update(1)
        else:
            result = {'id': self.request_id, 'ground_truth': self.request_truth, 'prompt': self.request_json, 'response': response}
            self.results_list.append(result)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            pbar.update(1)

def write_file(results_list, results_json_file):
    with open(results_json_file, "w") as f:
        json.dump(results_list, f, indent=4)

# import os
# import json
# import time
# import logging
# import aiohttp
# import asyncio
# from dataclasses import dataclass, field
# from tqdm import tqdm
# from src.tokens import num_tokens_from_messages

# logger = logging.getLogger(__name__)

# # ---------------------------
# # ASYNC API REQUEST MANAGER
# # ---------------------------
# async def async_api_requests(
#     max_requests_per_minute: float,
#     max_tokens_per_minute: float,
#     request_url: str,
#     api_key: str, 
#     root_path: str,
#     result_file_path: str, 
#     result_file_name: str, 
#     task: str,
#     dataset: str, 
#     model: str = 'gpt-4o-mini',
#     dataNum: int = 0, 
#     testNum: int = 1, 
#     method: str = 'base', 
#     max_token: int = 8000, 
#     max_attempts: int = 10,
#     temperature: float = 0,
#     choices: int = 1,
#     data=None, 
# ):
#     # ---------------------------
#     # Constants
#     # ---------------------------
#     seconds_to_pause_after_rate_limit_error = 15
#     seconds_to_sleep_each_loop = 0.01
#     checkpoint_interval = 1000
#     last_saved_count = 0

#     request_header = {"Authorization": f"Bearer {api_key}"}
#     queue_of_requests_to_retry = asyncio.Queue()
#     status_tracker = StatusTracker()
#     next_request = None

#     available_request_capacity = max_requests_per_minute
#     available_token_capacity = max_tokens_per_minute
#     last_update_time = time.time()

#     not_finished = True

#     if not os.path.exists(result_file_path):
#         os.makedirs(result_file_path)
#     results_json_file = os.path.join(result_file_path, result_file_name + ".json")
#     results_list = []

#     logging.basicConfig(
#         format='%(asctime)s %(message)s',
#         filename=os.path.join(result_file_path, result_file_name + ".log"),
#         encoding='utf-8',
#         level='WARNING'
#     )

#     testNum = min(testNum, len(data))
#     global pbar
#     pbar = tqdm(total=testNum - dataNum)

#     # ---------------------------
#     # Use a shared aiohttp session
#     # ---------------------------
#     async with aiohttp.ClientSession() as session:
#         while True:
#             # ---------------------------
#             # Prepare next request
#             # ---------------------------
#             if next_request is None:
#                 if not queue_of_requests_to_retry.empty():
#                     next_request = queue_of_requests_to_retry.get_nowait()
#                 elif not_finished:
#                     if dataNum < testNum:
#                         request_id = data[dataNum]['id']
#                         messages = data[dataNum]['prompt']
#                         request_truth = data[dataNum]['ground_truth']

#                         request_json = {
#                             "model": model,
#                             "messages": messages,
#                             "temperature": temperature,
#                             "top_p": 1,
#                             "n": choices,
#                             "stream": False,
#                         }

#                         next_request = APIRequest(
#                             request_id=request_id,
#                             request_json=request_json,
#                             request_truth=request_truth,
#                             token_consumption=num_tokens_from_messages(messages, model),
#                             attempts_left=max_attempts,
#                             metadata=request_json.pop("metadata", None),
#                             results_list=results_list,
#                         )

#                         status_tracker.num_tasks_started += 1
#                         status_tracker.num_tasks_in_progress += 1
#                         dataNum += 1
#                     else:
#                         not_finished = False

#             # ---------------------------
#             # Token & Request capacity control
#             # ---------------------------
#             current_time = time.time()
#             seconds_since_update = current_time - last_update_time
#             available_request_capacity = min(
#                 available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
#                 max_requests_per_minute
#             )
#             available_token_capacity = min(
#                 available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
#                 max_tokens_per_minute
#             )
#             last_update_time = current_time

#             # ---------------------------
#             # Dispatch next request
#             # ---------------------------
#             if next_request:
#                 next_request_tokens = next_request.token_consumption
#                 if available_request_capacity >= 1 and available_token_capacity >= next_request_tokens:
#                     available_request_capacity -= 1
#                     available_token_capacity -= next_request_tokens
#                     next_request.attempts_left -= 1

#                     asyncio.create_task(
#                         next_request.call_api(
#                             session=session,
#                             request_url=request_url,
#                             request_header=request_header,
#                             retry_queue=queue_of_requests_to_retry,
#                             save_filepath=results_json_file,
#                             status_tracker=status_tracker,
#                         )
#                     )
#                     next_request = None

#             # ---------------------------
#             # Periodic checkpoint save
#             # ---------------------------
#             current_count = len(results_list)
#             if current_count - last_saved_count >= checkpoint_interval:
#                 write_file(results_list, results_json_file)
#                 last_saved_count = current_count
#                 tqdm.write(f"✅ Checkpoint saved: {current_count} items processed.")

#             # ---------------------------
#             # Exit when done
#             # ---------------------------
#             if status_tracker.num_tasks_in_progress == 0 and not not_finished:
#                 break

#             # ---------------------------
#             # Sleep & Rate limit pause
#             # ---------------------------
#             await asyncio.sleep(seconds_to_sleep_each_loop)
#             seconds_since_rate_limit_error = (
#                 time.time() - status_tracker.time_of_last_rate_limit_error
#             )
#             if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
#                 tqdm.write("⚠️ Rate limit hit, sleeping temporarily...")
#                 await asyncio.sleep(seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)

#     # ---------------------------
#     # Final Save
#     # ---------------------------
#     write_file(results_list, results_json_file)
#     pbar.close()
#     tqdm.write("✅ Final save complete.")


# # ---------------------------
# # STATUS TRACKER
# # ---------------------------
# @dataclass
# class StatusTracker:
#     num_tasks_started: int = 0
#     num_tasks_in_progress: int = 0
#     num_tasks_succeeded: int = 0
#     num_tasks_failed: int = 0
#     num_rate_limit_errors: int = 0
#     num_api_errors: int = 0
#     num_other_errors: int = 0
#     time_of_last_rate_limit_error: int = 0


# # ---------------------------
# # API REQUEST OBJECT
# # ---------------------------
# @dataclass
# class APIRequest:
#     request_id: int
#     request_json: dict
#     request_truth: str
#     token_consumption: int
#     attempts_left: int
#     metadata: dict
#     results_list: list
#     result: list = field(default_factory=list)

#     async def call_api(self, session, request_url, request_header, retry_queue, save_filepath, status_tracker):
#         error = None
#         try:
#             async with session.post(url=request_url, headers=request_header, json=self.request_json) as response_raw:
#                 text = await response_raw.text()
#                 try:
#                     response = json.loads(text)
#                 except json.JSONDecodeError:
#                     response = {"error": {"message": f"Non-JSON response: {text[:200]}"}}

#             if "error" in response:
#                 status_tracker.num_api_errors += 1
#                 error = response
#                 if "Rate limit" in response["error"].get("message", ""):
#                     status_tracker.time_of_last_rate_limit_error = time.time()
#                     status_tracker.num_rate_limit_errors += 1
#                     status_tracker.num_api_errors -= 1

#         except Exception as e:
#             status_tracker.num_other_errors += 1
#             error = str(e)

#         # ---------------------------
#         # Handle result
#         # ---------------------------
#         if error:
#             self.result.append(error)
#             if self.attempts_left > 0:
#                 retry_queue.put_nowait(self)
#             else:
#                 result = {
#                     'id': self.request_id,
#                     'ground_truth': self.request_truth,
#                     'prompt': self.request_json,
#                     'response': str(error)
#                 }
#                 self.results_list.append(result)
#                 status_tracker.num_tasks_in_progress -= 1
#                 status_tracker.num_tasks_failed += 1
#                 pbar.update(1)
#                 tqdm.write(f"❌ Failed ID {self.request_id}: {error}")
#         else:
#             result = {
#                 'id': self.request_id,
#                 'ground_truth': self.request_truth,
#                 'prompt': self.request_json,
#                 'response': response
#             }
#             self.results_list.append(result)
#             status_tracker.num_tasks_in_progress -= 1
#             status_tracker.num_tasks_succeeded += 1
#             pbar.update(1)
#             if len(self.results_list) % 500 == 0:
#                 tqdm.write(f"✅ Progress: {len(self.results_list)} items completed.")


# # ---------------------------
# # FILE WRITER
# # ---------------------------
# def write_file(results_list, results_json_file):
#     with open(results_json_file, "w") as f:
#         json.dump(results_list, f, indent=4)
